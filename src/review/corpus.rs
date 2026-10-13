use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Component, Path},
    process::{Command, Stdio},
    sync::LazyLock,
};

const MAX_FILE_BYTES: usize = 256 * 1024;
const MAX_CORPUS_BYTES: usize = 16 * 1024 * 1024;
const MAX_CORPUS_FILES: usize = 10_000;

static SECRET_PATTERNS: LazyLock<[Regex; 4]> = LazyLock::new(|| {
    [
        Regex::new(r"(?im)-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")
            .expect("fixed private-key pattern is valid"),
        Regex::new(
            r"(?i)\b(?:gh[pousr]_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,}|sk-(?:proj-)?[a-z0-9_-]{20,}|xox[baprs]-[a-z0-9-]{20,}|akia[0-9a-z]{16}|eyj[a-z0-9_-]{15,}\.[a-z0-9_-]{15,}\.[a-z0-9_-]{15,})\b",
        )
        .expect("fixed token pattern is valid"),
        Regex::new(
            r#"(?im)(?:^|[^\w])["']?(?:api[_-]?key|client[_-]?secret|access[_-]?token|refresh[_-]?token|password|secret|credential)["']?\s*[:=]\s*["']?[A-Za-z0-9_./+=-]{24,}"#,
        )
        .expect("fixed labelled-secret pattern is valid"),
        Regex::new(
            r#"(?im)\b(?:authorization|proxy-authorization)\b\s*[:=]\s*(?:bearer|basic)\s+[A-Za-z0-9._~+/\-=]{24,}"#,
        )
        .expect("fixed authorization pattern is valid"),
    ]
});

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDocument {
    pub path: String,
    pub text: String,
}

pub fn load_project_corpus(project_root: &Path) -> Result<Vec<SourceDocument>> {
    let root = project_root.canonicalize().with_context(|| {
        format!(
            "Не удалось открыть корень проекта {}.",
            project_root.display()
        )
    })?;
    if !root.is_dir() {
        bail!("Корень проекта не является директорией.");
    }

    let candidates = list_tracked_regular_paths(&root)?
        .into_iter()
        .filter(|path| !is_excluded_path(path))
        .collect::<Vec<_>>();
    if candidates.len() > MAX_CORPUS_FILES {
        bail!("В проекте слишком много допустимых tracked-файлов для безопасной индексации.");
    }
    let ignored = ignored_paths(&root, &candidates)?;

    let mut documents = Vec::new();
    let mut total_bytes = 0usize;
    for path in candidates {
        if ignored.contains(&path) {
            continue;
        }

        let absolute_path = root.join(&path);
        let metadata = match fs::symlink_metadata(&absolute_path) {
            Ok(metadata) if metadata.file_type().is_file() => metadata,
            Ok(_) => continue,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error).with_context(|| format!("Не удалось проверить файл {path}."));
            }
        };
        if metadata.len() > MAX_FILE_BYTES as u64 {
            continue;
        }

        let canonical_path = match absolute_path.canonicalize() {
            Ok(path) if path.starts_with(&root) => path,
            Ok(_) => continue,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error).with_context(|| format!("Не удалось проверить файл {path}."));
            }
        };
        if !fs::symlink_metadata(&canonical_path)?.file_type().is_file() {
            continue;
        }

        let bytes = fs::read(&canonical_path)
            .with_context(|| format!("Не удалось прочитать tracked-файл {path}."))?;
        if bytes.len() > MAX_FILE_BYTES || bytes.contains(&0) {
            continue;
        }
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        if text.is_empty() || contains_secret_like_value(&text) {
            continue;
        }

        total_bytes = total_bytes.saturating_add(text.len());
        if total_bytes > MAX_CORPUS_BYTES {
            bail!("Допустимый текстовый corpus превышает лимит 16 MiB.");
        }
        documents.push(SourceDocument { path, text });
    }

    documents.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(documents)
}

fn list_tracked_regular_paths(root: &Path) -> Result<Vec<String>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "--stage", "-z"])
        .output()
        .context("Не удалось запустить git ls-files для построения project corpus.")?;
    if !output.status.success() {
        bail!("git ls-files не смог перечислить tracked-файлы проекта.");
    }

    let mut paths = Vec::new();
    for record in output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|record| !record.is_empty())
    {
        let Some(separator) = record.iter().position(|byte| *byte == b'\t') else {
            continue;
        };
        let Ok(header) = std::str::from_utf8(&record[..separator]) else {
            continue;
        };
        let mut fields = header.split_ascii_whitespace();
        let Some(mode) = fields.next() else { continue };
        let Some(_object_id) = fields.next() else {
            continue;
        };
        let Some(stage) = fields.next() else { continue };
        if stage != "0" || (mode != "100644" && mode != "100755") {
            continue;
        }

        let Ok(path) = std::str::from_utf8(&record[separator + 1..]) else {
            continue;
        };
        if is_safe_relative_path(path) && !path.chars().any(char::is_control) {
            paths.push(path.to_owned());
        }
    }
    paths.sort_unstable();
    paths.dedup();
    Ok(paths)
}

fn ignored_paths(root: &Path, paths: &[String]) -> Result<std::collections::HashSet<String>> {
    if paths.is_empty() {
        return Ok(Default::default());
    }

    let mut child = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["check-ignore", "--no-index", "--stdin", "-z"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("Не удалось запустить git check-ignore для project corpus.")?;
    let stdin = child
        .stdin
        .take()
        .context("Не удалось передать tracked paths в git check-ignore.")?;
    let output = std::thread::scope(|scope| -> Result<_> {
        let writer = scope.spawn(move || -> std::io::Result<()> {
            let mut stdin = stdin;
            for path in paths {
                stdin.write_all(path.as_bytes())?;
                stdin.write_all(&[0])?;
            }
            Ok(())
        });
        let output = child.wait_with_output()?;
        writer
            .join()
            .map_err(|_| anyhow::anyhow!("Поток ввода git check-ignore аварийно завершился."))?
            .context("Не удалось передать tracked paths в git check-ignore.")?;
        Ok(output)
    })?;
    if !output.status.success() && output.status.code() != Some(1) {
        bail!("git check-ignore не смог применить правила исключения проекта.");
    }

    Ok(output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .filter_map(|path| std::str::from_utf8(path).ok().map(str::to_owned))
        .collect())
}

fn is_safe_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !Path::new(path).is_absolute()
        && Path::new(path)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn is_excluded_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let Some(file_name) = lower.rsplit('/').next() else {
        return true;
    };

    let excluded_directory = lower.split('/').any(|component| {
        matches!(
            component,
            ".git"
                | ".gradle"
                | ".kotlin"
                | ".idea"
                | "build"
                | "coverage"
                | "dist"
                | "generated"
                | "node_modules"
                | "out"
                | "target"
                | "vendor"
        )
    });
    let secret_path = file_name.starts_with(".env")
        || file_name.split(['.', '_', '-']).any(|segment| {
            matches!(
                segment,
                "secret"
                    | "secrets"
                    | "credential"
                    | "credentials"
                    | "token"
                    | "tokens"
                    | "password"
                    | "passwords"
                    | "private"
            )
        })
        || ["private_key", "id_rsa", "id_ed25519", "service-account"]
            .iter()
            .any(|marker| file_name.contains(marker))
        || matches!(
            Path::new(file_name)
                .extension()
                .and_then(|extension| extension.to_str()),
            Some("pem" | "key" | "p12" | "pfx" | "jks" | "der" | "crt" | "cer" | "keystore")
        );
    let generated_file = file_name.ends_with(".min.js")
        || file_name.ends_with(".min.css")
        || file_name.ends_with(".map")
        || matches!(
            file_name,
            "cargo.lock" | "gradle.lockfile" | "package-lock.json" | "pnpm-lock.yaml" | "yarn.lock"
        );
    excluded_directory || secret_path || generated_file
}

fn contains_secret_like_value(text: &str) -> bool {
    SECRET_PATTERNS.iter().any(|pattern| pattern.is_match(text))
}

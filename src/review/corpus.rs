use crate::git_process::isolate_std_git_environment;
use anyhow::{Context, Result, bail};
use regex::Regex;
use same_file::Handle;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{Read, Write},
    path::{Component, Path},
    process::{Command, Stdio},
    sync::LazyLock,
};

const MAX_CORPUS_FILES: usize = 10_000;

static SECRET_PATTERNS: LazyLock<[Regex; 4]> = LazyLock::new(|| {
    [
        Regex::new(r"(?im)-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")
            .expect("fixed private-key pattern is valid"),
        Regex::new(
            r"(?i)\b(?:AIza[a-z0-9_-]{20,}|gh[pousr]_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,}|sk-(?:proj-)?[a-z0-9_-]{20,}|xox[baprs]-[a-z0-9-]{20,}|akia[0-9a-z]{16}|eyj[a-z0-9_-]{15,}\.[a-z0-9_-]{15,}\.[a-z0-9_-]{15,})\b",
        )
        .expect("fixed token pattern is valid"),
        Regex::new(
            r#"(?im)(?:^|[^\w])["']?(?:[\w]+[_-])*(?:api[_-]?key|client[_-]?secret|access[_-]?token|refresh[_-]?token|password|secret|credential)["']?\s*[:=]\s*["']?[A-Za-z0-9_./+=-]{24,}"#,
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CorpusDiagnostics {
    pub tracked_regular_files: usize,
    pub path_filtered_files: usize,
    pub git_ignored_files: usize,
    pub non_regular_files: usize,
    pub missing_files: usize,
    pub outside_root_files: usize,
    pub binary_files: usize,
    pub non_utf8_files: usize,
    pub empty_files: usize,
    pub secret_content_files: usize,
    pub safe_skipped_paths: Vec<String>,
}

impl CorpusDiagnostics {
    pub fn skipped_files(&self) -> usize {
        self.path_filtered_files
            + self.git_ignored_files
            + self.non_regular_files
            + self.missing_files
            + self.outside_root_files
            + self.binary_files
            + self.non_utf8_files
            + self.empty_files
            + self.secret_content_files
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadedCorpus {
    pub documents: Vec<SourceDocument>,
    pub diagnostics: CorpusDiagnostics,
}

pub fn load_project_corpus(project_root: &Path) -> Result<Vec<SourceDocument>> {
    Ok(load_project_corpus_with_diagnostics(project_root)?.documents)
}

pub fn load_project_corpus_with_diagnostics(project_root: &Path) -> Result<LoadedCorpus> {
    let root = project_root.canonicalize().with_context(|| {
        format!(
            "Не удалось открыть корень проекта {}.",
            project_root.display()
        )
    })?;
    if !root.is_dir() {
        bail!("Корень проекта не является директорией.");
    }

    let regular_paths = list_tracked_regular_paths(&root)?;
    let mut diagnostics = CorpusDiagnostics {
        tracked_regular_files: regular_paths.len(),
        ..CorpusDiagnostics::default()
    };
    let mut candidates = Vec::with_capacity(regular_paths.len());
    for path in regular_paths {
        if is_excluded_path(&path) {
            diagnostics.path_filtered_files += 1;
        } else {
            candidates.push(path);
        }
    }
    if candidates.len() > MAX_CORPUS_FILES {
        bail!("В проекте слишком много допустимых tracked-файлов для безопасной индексации.");
    }
    let ignored = ignored_paths(&root, &candidates)?;
    diagnostics.git_ignored_files = ignored.len();

    let mut documents = Vec::new();
    for path in candidates {
        if ignored.contains(&path) {
            continue;
        }

        let absolute_path = root.join(&path);
        match fs::symlink_metadata(&absolute_path) {
            Ok(metadata) if metadata.file_type().is_file() => (),
            Ok(_) => {
                diagnostics.non_regular_files += 1;
                record_safe_skipped_path(&mut diagnostics, &path);
                continue;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                diagnostics.missing_files += 1;
                record_safe_skipped_path(&mut diagnostics, &path);
                continue;
            }
            Err(error) => {
                return Err(error).with_context(|| format!("Не удалось проверить файл {path}."));
            }
        };

        let mut open_options = fs::OpenOptions::new();
        open_options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            open_options.custom_flags(libc::O_NONBLOCK);
        }
        let mut file = match open_options.open(&absolute_path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                diagnostics.missing_files += 1;
                record_safe_skipped_path(&mut diagnostics, &path);
                continue;
            }
            Err(error) => {
                return Err(error).with_context(|| format!("Не удалось открыть файл {path}."));
            }
        };
        if !file.metadata()?.is_file() {
            diagnostics.non_regular_files += 1;
            record_safe_skipped_path(&mut diagnostics, &path);
            continue;
        }

        let canonical_path = match absolute_path.canonicalize() {
            Ok(path) if path.starts_with(&root) => path,
            Ok(_) => {
                diagnostics.outside_root_files += 1;
                record_safe_skipped_path(&mut diagnostics, &path);
                continue;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                diagnostics.missing_files += 1;
                record_safe_skipped_path(&mut diagnostics, &path);
                continue;
            }
            Err(error) => {
                return Err(error).with_context(|| format!("Не удалось проверить файл {path}."));
            }
        };
        let opened_handle = Handle::from_file(
            file.try_clone()
                .with_context(|| format!("Не удалось проверить открытый файл {path}."))?,
        )
        .with_context(|| format!("Не удалось проверить открытый файл {path}."))?;
        let path_handle = match Handle::from_path(&canonical_path) {
            Ok(handle) => handle,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                diagnostics.missing_files += 1;
                record_safe_skipped_path(&mut diagnostics, &path);
                continue;
            }
            Err(error) => {
                return Err(error).with_context(|| format!("Не удалось проверить путь {path}."));
            }
        };
        if opened_handle != path_handle {
            diagnostics.outside_root_files += 1;
            record_safe_skipped_path(&mut diagnostics, &path);
            continue;
        }

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .with_context(|| format!("Не удалось прочитать tracked-файл {path}."))?;
        if bytes.contains(&0) {
            diagnostics.binary_files += 1;
            record_safe_skipped_path(&mut diagnostics, &path);
            continue;
        }
        let text = match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(_) => {
                diagnostics.non_utf8_files += 1;
                record_safe_skipped_path(&mut diagnostics, &path);
                continue;
            }
        };
        if text.is_empty() {
            diagnostics.empty_files += 1;
            record_safe_skipped_path(&mut diagnostics, &path);
            continue;
        }
        if contains_secret_like_value(&text) {
            diagnostics.secret_content_files += 1;
            continue;
        }

        documents.push(SourceDocument { path, text });
    }

    documents.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(LoadedCorpus {
        documents,
        diagnostics,
    })
}

fn record_safe_skipped_path(diagnostics: &mut CorpusDiagnostics, path: &str) {
    if diagnostics.safe_skipped_paths.len() < 10 && !is_excluded_path(path) {
        diagnostics.safe_skipped_paths.push(path.to_owned());
    }
}

fn list_tracked_regular_paths(root: &Path) -> Result<Vec<String>> {
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(root)
        .args(["ls-files", "--stage", "-z"]);
    isolate_std_git_environment(&mut command);
    let output = command
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

    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(root)
        .args(["check-ignore", "--no-index", "--stdin", "-z"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    isolate_std_git_environment(&mut command);
    let mut child = command
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

pub(crate) fn is_excluded_path(path: &str) -> bool {
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

pub(crate) fn contains_secret_like_value(text: &str) -> bool {
    SECRET_PATTERNS.iter().any(|pattern| pattern.is_match(text))
}

#[cfg(test)]
mod tests {
    use super::{MAX_CORPUS_FILES, load_project_corpus_with_diagnostics};
    use std::{fs, process::Command};
    use tempfile::tempdir;

    #[test]
    fn corpus_loader_accepts_10k_candidate_files_and_rejects_10k_plus_one() {
        let fixture = tempdir().expect("temporary Git project is created");
        let root = fixture.path();
        let initialize = Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(root)
            .status()
            .expect("Git initializes the fixture");
        assert!(initialize.success());

        for index in 0..=MAX_CORPUS_FILES {
            fs::write(root.join(format!("file-{index:05}.txt")), "fixture\n")
                .expect("tracked candidate is written");
        }
        let add = Command::new("git")
            .args(["add", "--all"])
            .current_dir(root)
            .status()
            .expect("Git stages the fixture candidates");
        assert!(add.success());

        let too_many = load_project_corpus_with_diagnostics(root)
            .expect_err("10 001 tracked candidates exceed the configured limit");
        assert!(too_many.to_string().contains("слишком много"));

        let last_file = format!("file-{:05}.txt", MAX_CORPUS_FILES);
        let remove = Command::new("git")
            .args(["rm", "--cached", "--quiet", "--", &last_file])
            .current_dir(root)
            .status()
            .expect("Git removes one fixture candidate from the index");
        assert!(remove.success());

        let corpus = load_project_corpus_with_diagnostics(root)
            .expect("10 000 tracked candidates remain within the limit");
        assert_eq!(corpus.diagnostics.tracked_regular_files, MAX_CORPUS_FILES);
        assert_eq!(corpus.documents.len(), MAX_CORPUS_FILES);
    }
}

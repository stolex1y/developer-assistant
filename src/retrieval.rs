use anyhow::{Context, Result, bail};
use regex::{Captures, Regex};
use std::{
    borrow::Cow,
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::LazyLock,
};

const MAX_CHUNK_CHARS: usize = 1_200;
const MAX_PASSAGES: usize = 5;
const MAX_CONTEXT_BYTES: usize = 8 * 1_024;

#[derive(Debug)]
pub struct Passage {
    pub source_path: String,
    pub heading: String,
    pub text: String,
    tokens: Vec<String>,
}

impl Passage {
    pub(crate) fn new(source_path: String, heading: String, text: String) -> Self {
        let mut tokens = tokenize(&heading);
        tokens.extend(tokenize(&text));
        Self {
            source_path,
            heading,
            text,
            tokens,
        }
    }
}

#[derive(Clone, Copy)]
pub struct RetrievedPassage<'a> {
    pub passage: &'a Passage,
    pub excerpt: &'a str,
}

pub fn load_corpus(project_root: &Path) -> Result<Vec<Passage>> {
    let mut paths = Vec::new();
    let readme = project_root.join("README.md");
    if is_regular_file(&readme)? {
        paths.push(readme);
    }

    let docs_dir = project_root.join("docs");
    match fs::symlink_metadata(&docs_dir) {
        Ok(metadata) if metadata.file_type().is_symlink() => {}
        Ok(metadata) if metadata.is_dir() => collect_markdown(&docs_dir, &mut paths)?,
        Ok(_) => bail!("Путь docs существует, но не является директорией."),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("Не удалось открыть {}.", docs_dir.display()));
        }
    }

    paths.sort();
    let mut passages = Vec::new();
    for path in paths {
        let source_path = path
            .strip_prefix(project_root)
            .context("Путь документа вышел за корень проекта.")?
            .to_string_lossy()
            .replace('\\', "/");
        let markdown = fs::read_to_string(&path)
            .with_context(|| format!("Не удалось прочитать документ {source_path}."))?;
        let redacted = redact_labelled_values(&markdown);
        parse_markdown(&source_path, &redacted, &mut passages);
    }
    Ok(passages)
}

pub fn retrieve<'a>(passages: &'a [Passage], question: &str) -> Vec<RetrievedPassage<'a>> {
    if passages.is_empty() {
        return Vec::new();
    }

    let mut query_terms = tokenize(question);
    query_terms.sort_unstable();
    query_terms.dedup();
    if query_terms.is_empty() {
        return Vec::new();
    }

    let term_frequencies: Vec<HashMap<&str, usize>> = passages
        .iter()
        .map(|passage| {
            let mut frequencies = HashMap::new();
            for token in &passage.tokens {
                *frequencies.entry(token.as_str()).or_insert(0) += 1;
            }
            frequencies
        })
        .collect();

    let mut document_frequencies = HashMap::new();
    for term in &query_terms {
        let frequency = term_frequencies
            .iter()
            .filter(|frequencies| frequencies.contains_key(term.as_str()))
            .count();
        document_frequencies.insert(term.as_str(), frequency);
    }

    let average_length = passages
        .iter()
        .map(|passage| passage.tokens.len())
        .sum::<usize>() as f64
        / passages.len() as f64;
    let average_length = average_length.max(1.0);
    let document_count = passages.len() as f64;
    let mut ranked = Vec::with_capacity(passages.len());

    for (index, passage) in passages.iter().enumerate() {
        let length = passage.tokens.len() as f64;
        let mut score = 0.0;
        for term in &query_terms {
            let Some(term_frequency) = term_frequencies[index].get(term.as_str()) else {
                continue;
            };
            let document_frequency = document_frequencies[term.as_str()] as f64;
            let inverse_document_frequency = (1.0
                + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
                .ln();
            let term_frequency = *term_frequency as f64;
            let normalization =
                term_frequency + 1.5 * (1.0 - 0.75 + 0.75 * length / average_length);
            score += inverse_document_frequency * term_frequency * 2.5 / normalization;
        }
        if score > 0.0 {
            ranked.push((index, score));
        }
    }

    ranked.sort_by(|(left_index, left_score), (right_index, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| {
                passages[*left_index]
                    .source_path
                    .cmp(&passages[*right_index].source_path)
            })
            .then_with(|| {
                passages[*left_index]
                    .heading
                    .cmp(&passages[*right_index].heading)
            })
    });

    let mut selected = Vec::with_capacity(MAX_PASSAGES.min(ranked.len()));
    let mut remaining_bytes = MAX_CONTEXT_BYTES;
    for (index, _) in ranked {
        if selected.len() == MAX_PASSAGES || remaining_bytes == 0 {
            break;
        }
        let passage = &passages[index];
        let excerpt = byte_prefix(&passage.text, remaining_bytes);
        if excerpt.is_empty() {
            continue;
        }
        remaining_bytes -= excerpt.len();
        selected.push(RetrievedPassage { passage, excerpt });
    }
    selected
}

pub fn redact_labelled_values(text: &str) -> Cow<'_, str> {
    static REDACTION_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r#"(?i)(^|[^\p{L}\p{N}])(api[\s_-]*key|api[\s_-]*token|access[\s_-]*token|refresh[\s_-]*token|client[\s_-]*secret|credentials?|token|secret|password|otp|phone|telephone|authorization|bearer|пароль|телефон|код|токен|секрет|авторизация)\b\s*[:=]\s*(?:`[^`\r\n]*`|"[^"\r\n]*"|'[^'\r\n]*'|“[^”\r\n]*”|‘[^’\r\n]*’|[^\r\n;]+)"#,
        )
        .expect("fixed credential redaction pattern is valid")
    });
    REDACTION_PATTERN.replace_all(text, |captures: &Captures<'_>| {
        format!("{}{} [REDACTED]", &captures[1], &captures[2])
    })
}

fn is_regular_file(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(metadata.file_type().is_file()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => {
            Err(error).with_context(|| format!("Не удалось проверить {}.", path.display()))
        }
    }
}

fn collect_markdown(directory: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(directory)
        .with_context(|| format!("Не удалось открыть {}.", directory.display()))?
    {
        entries.push(entry?.path());
    }
    entries.sort();

    for path in entries {
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("Не удалось проверить {}.", path.display()))?;
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            collect_markdown(&path, output)?;
        } else if metadata.is_file()
            && path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
        {
            output.push(path);
        }
    }
    Ok(())
}

fn parse_markdown(source_path: &str, markdown: &str, output: &mut Vec<Passage>) {
    let mut heading = "Общий раздел".to_owned();
    let mut section = String::new();
    let mut fence: Option<&str> = None;

    for line in markdown.lines() {
        let trimmed = line.trim_start();
        if let Some(marker) = fence {
            section.push_str(line);
            section.push('\n');
            if trimmed.starts_with(marker) {
                fence = None;
            }
            continue;
        }
        let marker = if trimmed.starts_with("```") {
            Some("```")
        } else if trimmed.starts_with("~~~") {
            Some("~~~")
        } else {
            None
        };
        if let Some(marker) = marker {
            fence = Some(marker);
            section.push_str(line);
            section.push('\n');
            continue;
        }
        if let Some(next_heading) = parse_heading(trimmed) {
            append_section(source_path, &heading, &section, output);
            section.clear();
            heading = next_heading;
        } else {
            section.push_str(line);
            section.push('\n');
        }
    }
    append_section(source_path, &heading, &section, output);
}

fn parse_heading(line: &str) -> Option<String> {
    let level = line
        .chars()
        .take_while(|character| *character == '#')
        .count();
    if !(1..=6).contains(&level) || !line.chars().nth(level).is_some_and(char::is_whitespace) {
        return None;
    }
    let heading = line[level..].trim();
    (!heading.is_empty()).then(|| heading.to_owned())
}

fn append_section(source_path: &str, heading: &str, section: &str, output: &mut Vec<Passage>) {
    let mut remaining = section.trim();
    while !remaining.is_empty() {
        let mut end = remaining
            .char_indices()
            .nth(MAX_CHUNK_CHARS)
            .map_or(remaining.len(), |(index, _)| index);
        if end < remaining.len()
            && let Some(boundary) = remaining[..end].rfind(char::is_whitespace)
            && boundary > 0
        {
            end = boundary;
        }

        let text = remaining[..end].trim();
        if !text.is_empty() {
            let passage = Passage::new(source_path.to_owned(), heading.to_owned(), text.to_owned());
            if !passage.tokens.is_empty() {
                output.push(passage);
            }
        }
        remaining = remaining[end..].trim_start();
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| token.chars().count() >= 2)
        .map(str::to_owned)
        .collect()
}

fn byte_prefix(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].trim_end()
}

#[cfg(test)]
mod tests {
    use super::{load_corpus, redact_labelled_values, retrieve};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn corpus_is_limited_to_readme_and_docs_markdown() {
        let root = tempdir().unwrap();
        fs::create_dir_all(root.path().join("docs/nested")).unwrap();
        fs::create_dir_all(root.path().join("config")).unwrap();
        fs::create_dir_all(root.path().join("examples")).unwrap();
        fs::write(
            root.path().join("README.md"),
            "# Overview\nProject summary.",
        )
        .unwrap();
        fs::write(
            root.path().join("docs/nested/architecture.md"),
            "# Modules\ncore web persistence",
        )
        .unwrap();
        fs::write(root.path().join("docs/notes.txt"), "not indexed").unwrap();
        fs::write(root.path().join("config/private.md"), "not indexed").unwrap();
        fs::write(root.path().join("examples/sample.md"), "not indexed").unwrap();

        let corpus = load_corpus(root.path()).unwrap();
        let paths: Vec<&str> = corpus
            .iter()
            .map(|passage| passage.source_path.as_str())
            .collect();
        assert_eq!(paths, ["README.md", "docs/nested/architecture.md"]);
        assert!(
            corpus
                .iter()
                .all(|passage| !passage.text.contains("not indexed"))
        );
    }

    #[test]
    fn bm25_returns_relevant_sources_and_obeys_context_budget() {
        let root = tempdir().unwrap();
        fs::create_dir_all(root.path().join("docs")).unwrap();
        fs::write(
            root.path().join("README.md"),
            "# Overview\nA project about finance.",
        )
        .unwrap();
        fs::write(
            root.path().join("docs/architecture.md"),
            format!(
                "# Modules\n{}",
                "core web persistence transport ".repeat(700)
            ),
        )
        .unwrap();
        let corpus = load_corpus(root.path()).unwrap();

        let retrieved = retrieve(&corpus, "модули core persistence transport");
        assert!(!retrieved.is_empty());
        assert_eq!(retrieved[0].passage.source_path, "docs/architecture.md");
        assert!(retrieved.len() <= 5);
        assert!(
            retrieved
                .iter()
                .map(|item| item.excerpt.len())
                .sum::<usize>()
                <= 8 * 1_024
        );
        assert!(retrieve(&corpus, "unmatched query tokens").is_empty());
    }

    #[test]
    fn labelled_credentials_are_redacted_before_indexing() {
        let redacted = redact_labelled_values(
            "phone: `+79990000000`, OTP: `000000` and password: `demo`; API_KEY=sk-test-value",
        );
        assert!(!redacted.contains("+79990000000"));
        assert!(!redacted.contains("000000"));
        assert!(!redacted.contains("demo"));
        assert!(!redacted.contains("sk-test-value"));
        assert!(redacted.contains("[REDACTED]"));
        let edge_cases = redact_labelled_values(
            "OPENAI_API_KEY=synthetic-key\npassword: пароль123\nsecret: alpha beta\nauthorization: Bearer alpha beta\nOTP: 123",
        );
        for credential in [
            "synthetic-key",
            "пароль123",
            "alpha beta",
            "Bearer alpha beta",
            "123",
        ] {
            assert!(!edge_cases.contains(credential));
        }
        assert_eq!(
            redact_labelled_values("API key is required").as_ref(),
            "API key is required"
        );
    }
}

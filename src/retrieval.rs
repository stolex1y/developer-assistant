use crate::review::corpus::{self, CorpusDiagnostics};
use anyhow::{Context, Result, bail};
use regex::{Captures, Regex};
use std::{
    borrow::Cow,
    collections::HashMap,
    ops::Range,
    path::{Component, Path, PathBuf},
    sync::LazyLock,
};

const MAX_CHUNK_CHARS: usize = 1_200;
pub(crate) const MAX_INITIAL_PASSAGES: usize = 5;
const MAX_TOOL_PASSAGES: usize = 8;
const MAX_CONTEXT_BYTES: usize = 8 * 1_024;
const MAX_READ_LINES: usize = 80;
const MAX_READ_BYTES: usize = 8 * 1_024;
const TRUNCATION_MARKER: &str = "[excerpt truncated at 8 KiB]\n";
const MAX_READ_CONTENT_BYTES: usize = MAX_READ_BYTES - TRUNCATION_MARKER.len();
const MAX_TOOL_PATH_BYTES: usize = 1_024;

#[derive(Debug)]
pub struct Passage {
    pub source_path: String,
    pub heading: String,
    pub start_line: usize,
    pub end_line: usize,
    text_range: Range<usize>,
    tokens: Vec<String>,
}

#[derive(Clone, Copy)]
pub struct RetrievedPassage<'a> {
    pub passage: &'a Passage,
    pub excerpt: &'a str,
}

impl RetrievedPassage<'_> {
    pub(crate) fn excerpt_end_line(&self) -> usize {
        self.passage.start_line + self.excerpt.lines().count().saturating_sub(1)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ReadFileExcerpt {
    pub text: String,
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Default)]
pub struct ProjectIndex {
    root: Option<PathBuf>,
    documents: HashMap<String, IndexedDocument>,
}

struct IndexedDocument {
    text: String,
    passages: Vec<Passage>,
    headings: Vec<HeadingAtLine>,
}

struct HeadingAtLine {
    start_line: usize,
    byte_range: Option<Range<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexRefreshSummary {
    pub indexed_files: usize,
    pub reused_files: usize,
    pub reindexed_files: usize,
    pub removed_files: usize,
    pub diagnostics: CorpusDiagnostics,
}

impl ProjectIndex {
    pub fn refresh(&mut self, project_root: &Path) -> Result<IndexRefreshSummary> {
        let root = project_root.canonicalize().with_context(|| {
            format!(
                "Не удалось открыть корень проекта {}.",
                project_root.display()
            )
        })?;
        let loaded = corpus::load_project_corpus_with_diagnostics(&root)?;
        let same_root = self.root.as_deref() == Some(root.as_path());
        let previous_count = self.documents.len();
        let mut previous = std::mem::take(&mut self.documents);
        if !same_root {
            previous.clear();
        }

        let mut documents = HashMap::with_capacity(loaded.documents.len());
        let mut reused_files = 0;
        let mut reindexed_files = 0;
        for source in loaded.documents {
            let sanitized = redact_labelled_values(&source.text);
            let existing = previous.remove(&source.path);
            let unchanged = existing
                .as_ref()
                .is_some_and(|indexed| indexed.text == sanitized.as_ref());
            let indexed = match existing {
                Some(indexed) if unchanged => {
                    reused_files += 1;
                    indexed
                }
                _ => {
                    reindexed_files += 1;
                    let text = sanitized.into_owned();
                    let mut headings = Vec::new();
                    let passages = parse_document(&source.path, &text, &mut headings);
                    IndexedDocument {
                        text,
                        passages,
                        headings,
                    }
                }
            };
            documents.insert(source.path, indexed);
        }

        let removed_files = if same_root {
            previous.len()
        } else {
            previous_count
        };
        let indexed_files = documents.len();
        self.root = Some(root);
        self.documents = documents;

        Ok(IndexRefreshSummary {
            indexed_files,
            reused_files,
            reindexed_files,
            removed_files,
            diagnostics: loaded.diagnostics,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<RetrievedPassage<'_>> {
        retrieve(self, query, limit)
    }

    pub(crate) fn heading_for_range(
        &self,
        path: &str,
        start_line: usize,
        _end_line: usize,
    ) -> Option<&str> {
        if !is_markdown_path(path) {
            return None;
        }
        let document = self.documents.get(path)?;
        let index = document
            .headings
            .partition_point(|heading| heading.start_line <= start_line);
        let heading = document.headings.get(index.checked_sub(1)?)?;
        match &heading.byte_range {
            Some(range) => document.text.get(range.clone()),
            None => Some("Общий раздел"),
        }
    }

    pub fn read_file(
        &self,
        path: &str,
        start_line: usize,
        end_line: usize,
    ) -> Result<ReadFileExcerpt> {
        if !is_safe_relative_path(path) || path.len() > MAX_TOOL_PATH_BYTES {
            bail!("read_file принимает только безопасный относительный путь из filtered corpus.");
        }
        if start_line == 0 || end_line < start_line || end_line - start_line >= MAX_READ_LINES {
            bail!("read_file требует диапазон от 1 до {MAX_READ_LINES} строк.");
        }
        let document = self
            .documents
            .get(path)
            .context("Путь отсутствует в filtered corpus.")?;
        let mut output = String::new();
        let mut actual_start_line = None;
        let mut actual_end_line = None;
        let mut truncated = false;
        for (index, line) in document.text.lines().enumerate() {
            let line_number = index + 1;
            if line_number < start_line || line_number > end_line {
                continue;
            }
            let prefix = format!("{line_number:>5} | ");
            let available = MAX_READ_CONTENT_BYTES.saturating_sub(output.len());
            if available < prefix.len() + 1 {
                truncated = true;
                break;
            }
            let remaining = available - prefix.len() - 1;
            let safe_line = utf8_prefix(line, remaining);
            output.push_str(&prefix);
            output.push_str(safe_line);
            output.push('\n');
            actual_start_line.get_or_insert(line_number);
            actual_end_line = Some(line_number);
            if safe_line.len() < line.len() {
                truncated = true;
                break;
            }
        }
        if truncated {
            output.push_str(TRUNCATION_MARKER);
        }
        if output.is_empty() {
            bail!("В файле нет запрошенного диапазона строк.");
        }
        Ok(ReadFileExcerpt {
            text: output,
            start_line: actual_start_line.context("В файле нет запрошенного диапазона строк.")?,
            end_line: actual_end_line.context("В файле нет запрошенного диапазона строк.")?,
        })
    }
}

pub fn retrieve<'a>(
    index: &'a ProjectIndex,
    question: &str,
    limit: usize,
) -> Vec<RetrievedPassage<'a>> {
    if limit == 0 {
        return Vec::new();
    }
    let passages = index
        .documents
        .values()
        .flat_map(|document| document.passages.iter())
        .collect::<Vec<_>>();
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
                    .start_line
                    .cmp(&passages[*right_index].start_line)
            })
            .then_with(|| {
                passages[*left_index]
                    .heading
                    .cmp(&passages[*right_index].heading)
            })
    });

    let mut selected = Vec::with_capacity(limit.min(MAX_TOOL_PASSAGES).min(ranked.len()));
    let mut remaining_bytes = MAX_CONTEXT_BYTES;
    for (passage_index, _) in ranked {
        if selected.len() == limit.min(MAX_TOOL_PASSAGES) || remaining_bytes == 0 {
            break;
        }
        let passage = passages[passage_index];
        let document = index
            .documents
            .get(&passage.source_path)
            .expect("indexed passage must reference an indexed document");
        let text = document
            .text
            .get(passage.text_range.clone())
            .expect("indexed passage range must remain within its document");
        let excerpt = byte_prefix(text, remaining_bytes);
        if excerpt.is_empty() {
            continue;
        }
        remaining_bytes -= excerpt.len();
        selected.push(RetrievedPassage { passage, excerpt });
    }
    selected
}

fn parse_document(
    source_path: &str,
    text: &str,
    headings: &mut Vec<HeadingAtLine>,
) -> Vec<Passage> {
    let mut passages = Vec::new();
    if is_markdown_path(source_path) {
        parse_markdown(source_path, text, &mut passages, headings);
    } else {
        parse_plain_text(source_path, text, &mut passages);
    }
    passages
}

fn is_markdown_path(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("md") || extension.eq_ignore_ascii_case("markdown")
        })
}

fn parse_plain_text(source_path: &str, text: &str, output: &mut Vec<Passage>) {
    let mut chunk = PassageChunk::default();
    let heading = "Файл";
    let mut offset = 0;
    for (index, raw_line) in text.split_inclusive('\n').enumerate() {
        let line_start = offset;
        let content = line_content(raw_line);
        let line_end = line_start + content.len();
        append_line(
            source_path,
            heading,
            text,
            line_start,
            line_end,
            index + 1,
            &mut chunk,
            output,
        );
        offset += raw_line.len();
    }
    flush_chunk(source_path, heading, text, &mut chunk, output);
}

fn parse_markdown(
    source_path: &str,
    text: &str,
    output: &mut Vec<Passage>,
    headings: &mut Vec<HeadingAtLine>,
) {
    let mut heading = "Общий раздел".to_owned();
    headings.push(HeadingAtLine {
        start_line: 1,
        byte_range: None,
    });
    let mut section_fence: Option<&str> = None;
    let mut chunk = PassageChunk::default();
    let mut offset = 0;

    for (index, raw_line) in text.split_inclusive('\n').enumerate() {
        let line_start = offset;
        let content = line_content(raw_line);
        let line_end = line_start + content.len();
        let trimmed = content.trim_start();

        if let Some(marker) = section_fence {
            append_line(
                source_path,
                &heading,
                text,
                line_start,
                line_end,
                index + 1,
                &mut chunk,
                output,
            );
            if trimmed.starts_with(marker) {
                section_fence = None;
            }
        } else if let Some(marker) = fence_marker(trimmed) {
            section_fence = Some(marker);
            append_line(
                source_path,
                &heading,
                text,
                line_start,
                line_end,
                index + 1,
                &mut chunk,
                output,
            );
        } else if let Some(next_heading) = parse_heading(trimmed) {
            flush_chunk(source_path, &heading, text, &mut chunk, output);
            let heading_start = next_heading.as_ptr() as usize - text.as_ptr() as usize;
            let heading_end = heading_start + next_heading.len();
            heading = next_heading.to_owned();
            headings.push(HeadingAtLine {
                start_line: index + 1,
                byte_range: Some(heading_start..heading_end),
            });
        } else {
            append_line(
                source_path,
                &heading,
                text,
                line_start,
                line_end,
                index + 1,
                &mut chunk,
                output,
            );
        }
        offset += raw_line.len();
    }
    flush_chunk(source_path, &heading, text, &mut chunk, output);
}

#[derive(Default)]
struct PassageChunk {
    start_offset: Option<usize>,
    end_offset: usize,
    start_line: usize,
    chars: usize,
}

#[allow(clippy::too_many_arguments)]
fn append_line(
    source_path: &str,
    heading: &str,
    source: &str,
    line_start: usize,
    line_end: usize,
    line_number: usize,
    chunk: &mut PassageChunk,
    output: &mut Vec<Passage>,
) {
    let line = &source[line_start..line_end];
    let line_chars = line.chars().count();
    if line_chars > MAX_CHUNK_CHARS {
        flush_chunk(source_path, heading, source, chunk, output);
        append_long_line(
            source_path,
            heading,
            source,
            line_start,
            line_end,
            line_number,
            output,
        );
        return;
    }

    if chunk.start_offset.is_some() && chunk.chars + line_chars + 1 > MAX_CHUNK_CHARS {
        flush_chunk(source_path, heading, source, chunk, output);
    }
    if chunk.start_offset.is_none() {
        chunk.start_offset = Some(line_start);
        chunk.start_line = line_number;
    }
    chunk.end_offset = line_end;
    chunk.chars += line_chars + 1;
}

fn append_long_line(
    source_path: &str,
    heading: &str,
    source: &str,
    line_start: usize,
    line_end: usize,
    line_number: usize,
    output: &mut Vec<Passage>,
) {
    let line = &source[line_start..line_end];
    let mut segment_start = line_start;
    let mut segment_chars = 0;
    for (byte_offset, _) in line.char_indices() {
        if segment_chars == MAX_CHUNK_CHARS {
            push_passage(
                source_path,
                heading,
                source,
                segment_start..line_start + byte_offset,
                line_number,
                output,
            );
            segment_start = line_start + byte_offset;
            segment_chars = 0;
        }
        segment_chars += 1;
    }
    push_passage(
        source_path,
        heading,
        source,
        segment_start..line_end,
        line_number,
        output,
    );
}

fn flush_chunk(
    source_path: &str,
    heading: &str,
    source: &str,
    chunk: &mut PassageChunk,
    output: &mut Vec<Passage>,
) {
    let Some(start_offset) = chunk.start_offset.take() else {
        return;
    };
    push_passage(
        source_path,
        heading,
        source,
        start_offset..chunk.end_offset,
        chunk.start_line,
        output,
    );
    *chunk = PassageChunk::default();
}

fn push_passage(
    source_path: &str,
    heading: &str,
    source: &str,
    byte_range: Range<usize>,
    start_line: usize,
    output: &mut Vec<Passage>,
) {
    let raw = &source[byte_range.clone()];
    let trimmed_start = raw.trim_start();
    let leading_bytes = raw.len() - trimmed_start.len();
    let leading_lines = raw[..leading_bytes]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count();
    let text = trimmed_start.trim_end();
    if text.is_empty() {
        return;
    }
    let start_line = start_line.saturating_add(leading_lines);
    let end_line = start_line.saturating_add(text.lines().count().saturating_sub(1));
    let range_start = byte_range.start + leading_bytes;
    let text_range = range_start..range_start + text.len();
    let mut tokens = Vec::new();
    tokenize_into(heading, &mut tokens);
    tokenize_into(text, &mut tokens);
    if tokens.is_empty() {
        return;
    }
    output.push(Passage {
        source_path: source_path.to_owned(),
        heading: heading.to_owned(),
        start_line,
        end_line,
        text_range,
        tokens,
    });
}

fn line_content(raw_line: &str) -> &str {
    raw_line
        .strip_suffix('\n')
        .unwrap_or(raw_line)
        .strip_suffix('\r')
        .unwrap_or_else(|| raw_line.strip_suffix('\n').unwrap_or(raw_line))
}

fn fence_marker(line: &str) -> Option<&'static str> {
    if line.starts_with("```") {
        Some("```")
    } else if line.starts_with("~~~") {
        Some("~~~")
    } else {
        None
    }
}

fn parse_heading(line: &str) -> Option<&str> {
    let level = line
        .chars()
        .take_while(|character| *character == '#')
        .count();
    if !(1..=6).contains(&level) || !line.chars().nth(level).is_some_and(char::is_whitespace) {
        return None;
    }
    let heading = line[level..].trim();
    (!heading.is_empty()).then_some(heading)
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    tokenize_into(text, &mut tokens);
    tokens
}

fn tokenize_into(text: &str, tokens: &mut Vec<String>) {
    let lowercase = text.to_lowercase();
    tokens.extend(
        lowercase
            .split(|character: char| !character.is_alphanumeric())
            .filter(|token| token.chars().count() >= 2)
            .map(str::to_owned),
    );
}

fn byte_prefix(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    utf8_prefix(text, max_bytes).trim_end()
}

fn utf8_prefix(text: &str, max_bytes: usize) -> &str {
    let mut end = text.len().min(max_bytes);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn is_safe_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !Path::new(path).is_absolute()
        && !path.chars().any(char::is_control)
        && Path::new(path)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
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

#[cfg(test)]
mod tests {
    use super::{MAX_READ_BYTES, ProjectIndex, TRUNCATION_MARKER, redact_labelled_values};
    use std::{fs, path::Path, process::Command};
    use tempfile::tempdir;

    #[test]
    fn index_uses_all_tracked_text_and_reports_safe_filter_diagnostics() {
        let root = tempdir().unwrap();
        fs::create_dir_all(root.path().join("docs")).unwrap();
        fs::create_dir_all(root.path().join("src")).unwrap();
        fs::create_dir_all(root.path().join("secrets")).unwrap();
        fs::create_dir_all(root.path().join("target")).unwrap();
        fs::write(root.path().join(".gitignore"), "ignored.txt\n").unwrap();
        fs::write(
            root.path().join("README.md"),
            "# Overview\ntransaction schema overview\n",
        )
        .unwrap();
        fs::write(
            root.path().join("docs/schema.md"),
            "# Schema\nunique_schema_marker v3\n",
        )
        .unwrap();
        fs::write(
            root.path().join("src/lib.rs"),
            "pub const ROUTE_MARKER: &str = \"/agent\";\n",
        )
        .unwrap();
        fs::write(root.path().join("ignored.txt"), "ignored tracked text\n").unwrap();
        fs::write(root.path().join("untracked.txt"), "untracked text\n").unwrap();
        fs::write(
            root.path().join("secrets/api-token.txt"),
            "synthetic secret path\n",
        )
        .unwrap();
        fs::write(
            root.path().join("docs/sensitive.txt"),
            "github token ghp_abcdefghijklmnopqrstuvwxyz1234567890\n",
        )
        .unwrap();
        fs::write(root.path().join("empty.txt"), "").unwrap();
        fs::write(root.path().join("binary.dat"), [b'x', 0, b'y']).unwrap();
        fs::write(root.path().join("invalid.txt"), [0xff, 0xfe]).unwrap();
        let mut large_spec = String::with_capacity(17 * 1024 * 1024 + 64);
        large_spec.push_str("LARGE_SPEC_MARKER\n");
        large_spec.extend(std::iter::repeat_n('x', 17 * 1024 * 1024));
        large_spec.push_str("\nLARGE_SPEC_END_MARKER\n");
        fs::write(root.path().join("docs/large-spec.md"), large_spec).unwrap();
        fs::write(
            root.path().join("target/generated.rs"),
            "generated source\n",
        )
        .unwrap();
        init_and_add(
            root.path(),
            &[
                ".gitignore",
                "README.md",
                "docs/schema.md",
                "src/lib.rs",
                "ignored.txt",
                "secrets/api-token.txt",
                "docs/sensitive.txt",
                "empty.txt",
                "binary.dat",
                "invalid.txt",
                "docs/large-spec.md",
                "target/generated.rs",
            ],
        );

        let mut index = ProjectIndex::default();
        let summary = index.refresh(root.path()).unwrap();
        let source = index.search("ROUTE_MARKER agent", 8);
        let docs = index.search("unique_schema_marker schema", 8);

        assert!(source.iter().any(|result| {
            result.passage.source_path == "src/lib.rs"
                && result.passage.start_line == 1
                && result.passage.end_line == 1
                && result.excerpt.contains("ROUTE_MARKER")
        }));
        assert!(docs.iter().any(|result| {
            result.passage.source_path == "docs/schema.md"
                && result.passage.heading == "Schema"
                && result.passage.start_line == 2
                && result.excerpt.contains("unique_schema_marker")
        }));

        let large = index.search("LARGE_SPEC_END_MARKER", 8);
        assert!(large.iter().any(|result| {
            result.passage.source_path == "docs/large-spec.md"
                && result.excerpt.contains("LARGE_SPEC_END_MARKER")
        }));

        assert!(index.read_file("ignored.txt", 1, 1).is_err());
        assert!(index.read_file("untracked.txt", 1, 1).is_err());
        assert!(index.read_file("secrets/api-token.txt", 1, 1).is_err());
        assert!(index.read_file("docs/sensitive.txt", 1, 1).is_err());
        assert_eq!(summary.diagnostics.git_ignored_files, 1);
        assert_eq!(summary.diagnostics.path_filtered_files, 2);
        assert_eq!(summary.diagnostics.secret_content_files, 1);
        assert_eq!(summary.diagnostics.empty_files, 1);
        assert_eq!(summary.diagnostics.binary_files, 1);
        assert_eq!(summary.diagnostics.non_utf8_files, 1);
        assert!(!summary.diagnostics.safe_skipped_paths.iter().any(
            |path| path.contains("api-token")
                || path.contains("sensitive")
                || path == "ignored.txt"
        ));
        assert_eq!(summary.diagnostics.skipped_files(), 7);
    }

    #[test]
    fn refresh_reuses_unchanged_files_and_updates_additions_changes_and_deletions() {
        let root = tempdir().unwrap();
        fs::create_dir_all(root.path().join("docs")).unwrap();
        fs::create_dir_all(root.path().join("src")).unwrap();
        fs::write(
            root.path().join("README.md"),
            "# Project\nunchanged persistence marker\n",
        )
        .unwrap();
        fs::write(
            root.path().join("docs/remove.md"),
            "# Old\nold_deletion_marker\n",
        )
        .unwrap();
        fs::write(
            root.path().join("src/lib.rs"),
            "const OLD_ROUTE_MARKER: &str = \"/old\";\n",
        )
        .unwrap();
        init_and_add(root.path(), &["README.md", "docs/remove.md", "src/lib.rs"]);

        let mut index = ProjectIndex::default();
        let first = index.refresh(root.path()).unwrap();
        assert_eq!(first.indexed_files, 3);
        assert_eq!(first.reindexed_files, 3);

        fs::write(
            root.path().join("src/lib.rs"),
            "const NEW_ROUTE_MARKER: &str = \"/new\";\n",
        )
        .unwrap();
        fs::write(
            root.path().join("src/new.rs"),
            "const ADDED_ROUTE_MARKER: bool = true;\n",
        )
        .unwrap();
        fs::remove_file(root.path().join("docs/remove.md")).unwrap();
        git(root.path(), &["add", "-A"]);

        let second = index.refresh(root.path()).unwrap();
        let new_results = index.search("NEW_ROUTE_MARKER", 8);
        assert!(
            !index
                .read_file("src/lib.rs", 1, 1)
                .unwrap()
                .text
                .contains("OLD_ROUTE_MARKER")
        );
        assert_eq!(second.indexed_files, 3);
        assert_eq!(second.reused_files, 1);
        assert_eq!(second.reindexed_files, 2);
        assert_eq!(second.removed_files, 1);
        assert!(new_results.iter().any(|result| {
            result.passage.source_path == "src/lib.rs"
                && result.excerpt.contains("NEW_ROUTE_MARKER")
        }));
        assert!(index.read_file("docs/remove.md", 1, 2).is_err());
    }

    #[test]
    fn refresh_drops_tracked_files_when_gitignore_changes() {
        let root = tempdir().unwrap();
        fs::create_dir_all(root.path().join("docs")).unwrap();
        fs::write(root.path().join("README.md"), "# Project\nOverview.\n").unwrap();
        fs::write(
            root.path().join("docs/session-notes.md"),
            "# Notes\nIGNORED_SESSION_MARKER\n",
        )
        .unwrap();
        init_and_add(root.path(), &["README.md", "docs/session-notes.md"]);

        let mut index = ProjectIndex::default();
        index.refresh(root.path()).unwrap();
        assert!(
            index
                .read_file("docs/session-notes.md", 2, 2)
                .unwrap()
                .text
                .contains("IGNORED_SESSION_MARKER")
        );

        fs::write(root.path().join(".gitignore"), "docs/session-notes.md\n").unwrap();
        git(root.path(), &["add", ".gitignore"]);

        let summary = index.refresh(root.path()).unwrap();
        assert_eq!(summary.diagnostics.git_ignored_files, 1);
        assert_eq!(summary.removed_files, 1);
        assert!(
            index
                .search("IGNORED_SESSION_MARKER", 8)
                .iter()
                .all(|result| result.passage.source_path != "docs/session-notes.md")
        );
        assert!(index.read_file("docs/session-notes.md", 2, 2).is_err());
    }

    #[test]
    fn read_file_reports_actual_range_and_caps_truncated_text() {
        let root = tempdir().unwrap();
        fs::create_dir_all(root.path().join("docs")).unwrap();
        fs::write(
            root.path().join("README.md"),
            "# Overview\nfirst line\nsecond line\n",
        )
        .unwrap();
        fs::write(
            root.path().join("docs/long.md"),
            format!("# Long\n{}\n", "é".repeat(10_000)),
        )
        .unwrap();
        init_and_add(root.path(), &["README.md", "docs/long.md"]);

        let mut index = ProjectIndex::default();
        index.refresh(root.path()).unwrap();

        let short = index.read_file("README.md", 1, 80).unwrap();
        assert_eq!((short.start_line, short.end_line), (1, 3));
        assert!(short.text.contains("second line"));

        let long = index.read_file("docs/long.md", 2, 2).unwrap();
        assert_eq!((long.start_line, long.end_line), (2, 2));
        assert!(long.text.len() <= MAX_READ_BYTES);
        assert!(long.text.ends_with(TRUNCATION_MARKER));
    }

    #[test]
    fn heading_lookup_covers_long_line_passages() {
        let root = tempdir().unwrap();
        fs::create_dir_all(root.path().join("docs")).unwrap();
        fs::write(
            root.path().join("docs/architecture.md"),
            format!("# Architecture\n{}\n", "\"".repeat(6_000)),
        )
        .unwrap();
        init_and_add(root.path(), &["docs/architecture.md"]);

        let mut index = ProjectIndex::default();
        index.refresh(root.path()).unwrap();

        let passage_ranges: Vec<_> = index.documents["docs/architecture.md"]
            .passages
            .iter()
            .map(|passage| {
                (
                    passage.start_line,
                    passage.end_line,
                    passage.heading.as_str(),
                )
            })
            .collect();
        assert_eq!(
            index.heading_for_range("docs/architecture.md", 2, 2),
            Some("Architecture"),
            "{passage_ranges:?}"
        );
    }

    #[test]
    fn retrieval_tracks_line_ranges_after_trimming_blank_lines() {
        let root = tempdir().unwrap();
        fs::write(
            root.path().join("README.md"),
            "# Architecture\n\n\nactual detail marker\n\n\n",
        )
        .unwrap();
        init_and_add(root.path(), &["README.md"]);

        let mut index = ProjectIndex::default();
        index.refresh(root.path()).unwrap();

        let result = index.search("actual detail marker", 1);
        assert_eq!(result.len(), 1);
        assert_eq!(
            (result[0].passage.start_line, result[0].passage.end_line),
            (4, 4)
        );
        assert_eq!(result[0].excerpt, "actual detail marker");
        assert_eq!(result[0].excerpt_end_line(), 4);
    }

    #[test]
    fn retrieval_scores_markdown_heading_terms_without_returning_heading_as_excerpt() {
        let root = tempdir().unwrap();
        fs::write(
            root.path().join("README.md"),
            "# Persistence Architecture\nThe module stores data on disk.\n",
        )
        .unwrap();
        init_and_add(root.path(), &["README.md"]);

        let mut index = ProjectIndex::default();
        index.refresh(root.path()).unwrap();

        let result = index.search("persistence", 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].passage.heading, "Persistence Architecture");
        assert_eq!(result[0].excerpt, "The module stores data on disk.");
    }

    #[test]
    fn retrieval_is_ranked_and_bounded_by_passages_and_bytes() {
        let root = tempdir().unwrap();
        fs::create_dir_all(root.path().join("docs")).unwrap();
        fs::write(
            root.path().join("README.md"),
            "# Overview\nfinance project summary\n",
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
        init_and_add(root.path(), &["README.md", "docs/architecture.md"]);

        let mut index = ProjectIndex::default();
        index.refresh(root.path()).unwrap();
        let retrieved = index.search("modules core persistence transport", 5);

        assert!(!retrieved.is_empty());
        assert_eq!(retrieved[0].passage.source_path, "docs/architecture.md");
        assert_eq!(retrieved[0].passage.heading, "Modules");
        assert!(retrieved.len() <= 5);
        assert!(
            retrieved
                .iter()
                .map(|item| item.excerpt.len())
                .sum::<usize>()
                <= 8 * 1_024
        );
        assert!(index.search("unmatched query tokens", 5).is_empty());
    }

    #[test]
    fn retrieved_passage_end_line_matches_budget_truncated_excerpt() {
        let root = tempdir().unwrap();
        let lines = (0..2_000)
            .map(|line| format!("budgetmarker line {line:04} details\n"))
            .collect::<String>();
        fs::write(root.path().join("README.md"), format!("# Large\n{lines}")).unwrap();
        init_and_add(root.path(), &["README.md"]);

        let mut index = ProjectIndex::default();
        index.refresh(root.path()).unwrap();
        let passages = index.search("budgetmarker", 8);
        let truncated = passages
            .iter()
            .find(|passage| passage.excerpt_end_line() < passage.passage.end_line)
            .expect("search budget truncates the final returned passage");

        assert_eq!(
            truncated.excerpt_end_line(),
            truncated.passage.start_line + truncated.excerpt.lines().count() - 1
        );
        assert!(truncated.excerpt_end_line() < truncated.passage.end_line);
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

    fn init_and_add(root: &Path, paths: &[&str]) {
        git(root, &["init", "--initial-branch=main"]);
        let mut args = vec!["add", "-f", "--"];
        args.extend_from_slice(paths);
        git(root, &args);
    }

    fn git(root: &Path, args: &[&str]) {
        let output = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .env("GIT_OPTIONAL_LOCKS", "0")
            .output()
            .expect("git is installed");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

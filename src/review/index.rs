use crate::review::{
    cache::prepare_cache_directory,
    corpus::{SourceDocument, load_project_corpus},
    embedding::{EMBEDDING_DIMENSIONS, Embedder, MODEL_ID},
    retrieval::{HybridRetriever, Passage, SearchHit, chunk_documents},
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const INDEX_SCHEMA_VERSION: u32 = 1;
const MAX_CACHE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_READ_LINES: usize = 80;
const MAX_READ_BYTES: usize = 8 * 1024;
static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Serialize, Deserialize)]
struct CachedIndex {
    schema_version: u32,
    model: String,
    corpus_fingerprint: String,
    passages: Vec<Passage>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RetrievedPassage {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub score: f64,
}

pub struct ReviewIndex<E> {
    documents: HashMap<String, SourceDocument>,
    passages: Vec<Passage>,
    retriever: HybridRetriever,
    embedder: E,
}

impl<E: Embedder> ReviewIndex<E> {
    pub fn open(project_root: &Path, cache_directory: &Path, mut embedder: E) -> Result<Self> {
        let root = project_root.canonicalize().with_context(|| {
            format!(
                "Не удалось открыть корень проекта {}.",
                project_root.display()
            )
        })?;
        let documents = load_project_corpus(&root)?;
        let expected_passages = chunk_documents(&documents);
        let corpus_fingerprint = corpus_fingerprint(&documents);

        let cache_directory = prepare_cache_directory(&root, cache_directory)?;
        let cache_path = cache_directory.join("review-index-v1.json");

        let cached_passages =
            load_cached_passages(&cache_path, &corpus_fingerprint, &expected_passages);
        let passages = if let Some(passages) = cached_passages {
            passages
        } else {
            let inputs = expected_passages
                .iter()
                .map(|passage| passage.text.as_str())
                .collect::<Vec<_>>();
            let vectors = embedder.embed(&inputs)?;
            if vectors.len() != expected_passages.len() {
                bail!("Embedding runtime вернул неверное число passage vectors.");
            }
            let mut passages = expected_passages;
            for (passage, vector) in passages.iter_mut().zip(vectors) {
                validate_embedding(&vector)?;
                passage.embedding = vector;
            }
            save_cache(&cache_path, &corpus_fingerprint, &passages)?;
            passages
        };

        let documents = documents
            .into_iter()
            .map(|document| (document.path.clone(), document))
            .collect();
        let retriever = HybridRetriever::new(&passages);
        Ok(Self {
            documents,
            passages,
            retriever,
            embedder,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.passages.is_empty()
    }

    pub fn search(&mut self, query: &str, limit: usize) -> Result<Vec<RetrievedPassage>> {
        if self.passages.is_empty() {
            return Ok(Vec::new());
        }
        let query = query.trim();
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let query_embedding = self.embedder.embed(&[query])?;
        let query_embedding = query_embedding
            .first()
            .context("Embedding runtime не вернул vector для search query.")?;
        validate_embedding(query_embedding)?;

        let hits = self
            .retriever
            .search(&self.passages, query, query_embedding, limit)?;
        Ok(hits
            .into_iter()
            .map(
                |SearchHit {
                     passage_index,
                     score,
                 }| {
                    let passage = &self.passages[passage_index];
                    RetrievedPassage {
                        path: passage.source_path.clone(),
                        start_line: passage.start_line,
                        end_line: passage.end_line,
                        text: passage.text.clone(),
                        score,
                    }
                },
            )
            .collect())
    }

    pub fn read_file(&self, path: &str, start_line: usize, end_line: usize) -> Result<String> {
        if !is_safe_relative_path(path) {
            bail!("read_file принимает только относительный путь из filtered corpus.");
        }
        if start_line == 0 || end_line < start_line {
            bail!("read_file требует корректный диапазон строк, начинающийся с 1.");
        }
        let requested_lines = end_line - start_line + 1;
        if requested_lines > MAX_READ_LINES {
            bail!("read_file разрешает не более {MAX_READ_LINES} строк за один вызов.");
        }
        let document = self
            .documents
            .get(path)
            .context("Путь отсутствует в filtered corpus.")?;
        let mut output = String::new();
        let mut truncated = false;
        for (index, line) in document.text.lines().enumerate() {
            let line_number = index + 1;
            if line_number < start_line || line_number > end_line {
                continue;
            }
            let prefix = format!("{line_number:>5} | ");
            let remaining = MAX_READ_BYTES.saturating_sub(output.len() + prefix.len());
            if remaining == 0 {
                truncated = true;
                break;
            }
            let original_line = line;
            let line = utf8_prefix(original_line, remaining);
            output.push_str(&prefix);
            output.push_str(line);
            output.push('\n');
            if line.len() < original_line.len() {
                truncated = true;
                break;
            }
        }
        if truncated {
            let marker = "[excerpt truncated at 8 KiB]\n";
            let keep_bytes = MAX_READ_BYTES.saturating_sub(marker.len());
            output.truncate(utf8_prefix(&output, keep_bytes).len());
            output.push_str(marker);
        }
        if output.is_empty() {
            bail!("В файле нет запрошенного диапазона строк.");
        }
        Ok(output)
    }
    pub(crate) fn contains_path(&self, path: &str) -> bool {
        self.documents.contains_key(path)
    }

    pub(crate) fn contains_evidence(&self, path: &str, evidence: &str) -> bool {
        !evidence.is_empty()
            && self
                .documents
                .get(path)
                .is_some_and(|document| document.text.contains(evidence))
    }
}

fn corpus_fingerprint(documents: &[SourceDocument]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(INDEX_SCHEMA_VERSION.to_be_bytes());
    hasher.update(MODEL_ID.as_bytes());
    for document in documents {
        hasher.update((document.path.len() as u64).to_be_bytes());
        hasher.update(document.path.as_bytes());
        hasher.update((document.text.len() as u64).to_be_bytes());
        hasher.update(document.text.as_bytes());
    }
    hex_digest(&hasher.finalize())
}

fn load_cached_passages(
    cache_path: &Path,
    corpus_fingerprint: &str,
    expected_passages: &[Passage],
) -> Option<Vec<Passage>> {
    let metadata = fs::symlink_metadata(cache_path).ok()?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_CACHE_BYTES {
        return None;
    }
    let bytes = fs::read(cache_path).ok()?;
    let cache: CachedIndex = serde_json::from_slice(&bytes).ok()?;
    if cache.schema_version != INDEX_SCHEMA_VERSION
        || cache.model != MODEL_ID
        || cache.corpus_fingerprint != corpus_fingerprint
        || cache.passages.len() != expected_passages.len()
    {
        return None;
    }
    if cache
        .passages
        .iter()
        .zip(expected_passages)
        .any(|(cached, expected)| {
            cached.source_path != expected.source_path
                || cached.start_line != expected.start_line
                || cached.end_line != expected.end_line
                || cached.text != expected.text
                || validate_embedding(&cached.embedding).is_err()
        })
    {
        return None;
    }
    Some(cache.passages)
}

fn save_cache(cache_path: &Path, corpus_fingerprint: &str, passages: &[Passage]) -> Result<()> {
    save_cache_with_limit(
        cache_path,
        corpus_fingerprint,
        passages,
        MAX_CACHE_BYTES,
        "256 MiB",
    )
}

fn save_cache_with_limit(
    cache_path: &Path,
    corpus_fingerprint: &str,
    passages: &[Passage],
    max_bytes: u64,
    max_bytes_label: &str,
) -> Result<()> {
    let cache = CachedIndex {
        schema_version: INDEX_SCHEMA_VERSION,
        model: MODEL_ID.to_owned(),
        corpus_fingerprint: corpus_fingerprint.to_owned(),
        passages: passages.to_vec(),
    };
    let bytes =
        serde_json::to_vec(&cache).context("Не удалось сериализовать локальный review index.")?;
    if bytes.len() as u64 > max_bytes {
        bail!("Сериализованный review index превышает лимит {max_bytes_label}.");
    }

    let parent = cache_path
        .parent()
        .context("У cache path отсутствует родительская директория.")?;
    let temp_path = temporary_cache_path(parent);
    let write_result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .context("Не удалось создать временный review index.")?;
        file.write_all(&bytes)
            .context("Не удалось записать временный review index.")?;
        file.sync_all()
            .context("Не удалось сбросить временный review index на диск.")?;
        fs::rename(&temp_path, cache_path)
            .context("Не удалось атомарно заменить локальный review index.")?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .context("Не удалось сбросить изменения index cache на диск.")?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    write_result
}

fn temporary_cache_path(parent: &Path) -> PathBuf {
    let sequence = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(
        ".review-index-{}-{sequence}.tmp",
        std::process::id()
    ))
}

fn validate_embedding(embedding: &[f32]) -> Result<()> {
    if embedding.len() != EMBEDDING_DIMENSIONS || embedding.iter().any(|value| !value.is_finite()) {
        bail!("Review index содержит vector неверного размера или с нечисловым значением.");
    }
    Ok(())
}

fn is_safe_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !Path::new(path).is_absolute()
        && Path::new(path)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn utf8_prefix(text: &str, max_bytes: usize) -> &str {
    let mut end = text.len().min(max_bytes);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn hex_digest(digest: &[u8]) -> String {
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{MAX_CACHE_BYTES, save_cache_with_limit};
    use tempfile::tempdir;

    #[test]
    fn serialized_cache_respects_limit() {
        assert_eq!(MAX_CACHE_BYTES, 256 * 1024 * 1024);
        let directory = tempdir().expect("temporary directory is created");
        let cache_path = directory.path().join("review-index.json");

        let error = save_cache_with_limit(&cache_path, "fixture", &[], 1, "test limit")
            .expect_err("oversized serialized cache is rejected");

        assert!(error.to_string().contains("test limit"));
        assert!(!cache_path.exists(), "oversized cache is not written");
    }
}

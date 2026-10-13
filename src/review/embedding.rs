use crate::review::cache::prepare_cache_directory;
use anyhow::{Context, Result, bail};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::path::Path;

pub const MODEL_ID: &str = "fastembed-6.1.0/Xenova-all-MiniLM-L6-v2-quantized-onnx";
pub const EMBEDDING_DIMENSIONS: usize = 384;

pub trait Embedder {
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

pub struct FastEmbedder {
    model: TextEmbedding,
}

impl FastEmbedder {
    pub fn load(project_root: &Path, requested_cache_dir: &Path) -> Result<Self> {
        let root = project_root
            .canonicalize()
            .context("Не удалось определить корень для embedding model cache.")?;
        let cache_dir = prepare_cache_directory(&root, requested_cache_dir)?;
        let options = TextInitOptions::new(EmbeddingModel::AllMiniLML6V2Q)
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(true)
            .with_intra_threads(2);
        let model = TextEmbedding::try_new(options)
            .context("Не удалось загрузить локальную ONNX embedding model.")?;
        Ok(Self { model })
    }
}

impl Embedder for FastEmbedder {
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let embeddings = self
            .model
            .embed(texts.to_vec(), None)
            .context("Не удалось вычислить локальные text embeddings.")?;
        if embeddings.len() != texts.len() {
            bail!("Embedding runtime вернул неверное число vectors.");
        }
        if embeddings.iter().any(|embedding| {
            embedding.len() != EMBEDDING_DIMENSIONS
                || embedding.iter().any(|value| !value.is_finite())
        }) {
            bail!("Embedding runtime вернул vector неверного размера или с нечисловым значением.");
        }
        Ok(embeddings)
    }
}

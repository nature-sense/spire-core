use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// The result of embedding a piece of text.
///
/// Mirrors the `Embedding` interface from the TypeScript `spire` project's
/// `memory.ts` (`IEmbedder` contract).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// 384-dimensional float vector (L2-normalized).
    pub vector: Vec<f32>,
    /// The original text that was embedded.
    pub text: String,
    /// MD5 hash of the text (for caching / deduplication).
    pub text_hash: String,
    /// Estimated token count (split by whitespace).
    pub token_count: usize,
    /// Dimensionality of the vector (always 384 for all-MiniLM-L6-v2).
    pub dimensions: usize,
    /// Model identifier, e.g. "sentence-transformers/all-MiniLM-L6-v2".
    pub model_name: String,
    /// Timestamp when this embedding was generated.
    pub generated_at: DateTime<Utc>,
}

impl Embedding {
    /// Create a new `Embedding` from a raw vector and its source text.
    pub fn new(vector: Vec<f32>, text: &str, model_name: &str) -> Self {
        use md5::{Digest, Md5};
        let text_hash = format!("{:x}", Md5::digest(text.as_bytes()));
        let token_count = text.split_whitespace().count();
        Self {
            dimensions: vector.len(),
            vector,
            text: text.to_owned(),
            text_hash,
            token_count,
            model_name: model_name.to_owned(),
            generated_at: Utc::now(),
        }
    }
}

/// Trait that any embedder implementation must satisfy.
///
/// This mirrors the `IEmbedder` interface from the TypeScript `spire` project.
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    /// Generate an embedding for a single text string.
    async fn embed(&self, text: &str) -> anyhow::Result<Embedding>;

    /// Generate embeddings for multiple texts in batch.
    async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Embedding>>;

    /// Return the dimensionality of the embedding vectors (384).
    fn dimensions(&self) -> usize;
}

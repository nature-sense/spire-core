// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

pub mod candle_embedder;

pub use candle_embedder::{create_embedder, CandleEmbedder};

use anyhow::{anyhow, Result};
use async_trait::async_trait;

use crate::models::embedding::{Embedder, Embedding};

/// Fallback embedder used when the real CandleEmbedder fails to load.
///
/// This NEVER returns vectors — every call fails loudly so RAG ingest/search
/// surfaces a clear error instead of silently degrading to zero-vector results.
#[derive(Debug)]
pub struct NoopEmbedder;

#[async_trait]
impl Embedder for NoopEmbedder {
    async fn embed(&self, text: &str) -> Result<Embedding> {
        Err(anyhow!(
            "Embedding model unavailable (no embedder registered) — cannot embed {:?}. RAG requires sentence-transformers/all-MiniLM-L6-v2 to load.",
            text.chars().take(80).collect::<String>()
        ))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        Err(anyhow!(
            "Embedding model unavailable (no embedder registered) — cannot embed {} text(s). RAG requires sentence-transformers/all-MiniLM-L6-v2 to load.",
            texts.len()
        ))
    }

    fn dimensions(&self) -> usize {
        0
    }
}

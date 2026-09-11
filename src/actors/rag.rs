// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! RAG actor — per-platform retrieval-augmented generation.
//!
//! Retrieval re-embeds the candidate chunk texts with the shared embedder and
//! scores with **cosine similarity** on the real vectors (the chunks were
//! embedded at ingest time). When the embedder is the no-op placeholder (zero
//! vectors), scoring falls back to lexical Jaccard overlap so the tool still
//! works in degraded mode.
//!
//! Ingest is driven by the canonical `ingest.yaml` GraphRAG config format
//! (see [`crate::actors::rag_ingest`]) — the legacy `RagManifest` /
//! `RagSourceConfig` shapes are retired.

use crate::actors::rag_ingest::{self, GraphRagConfig, IngestReport};
use crate::actors::Actor;
use crate::config::knowledge_dir;
use crate::embedder::NoopEmbedder;
use crate::models::embedding::Embedder;
use crate::subsystems::graph::memory_graph::MemoryGraphMessage;
use anyhow::Result;
use async_trait::async_trait;
use spire_actor::registry::ServiceRegistry;
use std::path::{Path, PathBuf};
use tokio::sync::{mpsc, oneshot};
use tracing::info;

// ============================================================================
// RAG types
// ============================================================================

/// A discoverable ingestion manifest (the "ingest script" the UI lists) —
/// parsed from the canonical `GraphRagConfig` `ingest.yaml`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RagManifestInfo {
    /// Corpus/domain the manifest ingests into (the RAG retrieval scope),
    /// resolved from the script itself (`pipeline.corpus`, else its
    /// `knowledge/<corpus>/ingest.yaml` directory name).
    pub domain: String,
    /// Absolute path to the `ingest.yaml` file.
    pub path: String,
    /// Resolved deterministic corpus fingerprint (16-hex).
    pub corpus_version: String,
    /// Human-readable description from the config (may be empty).
    pub description: String,
}

/// A per-domain summary of the ingested corpus ("state of the RAG data").
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RagDomainInfo {
    /// Clean domain id (e.g. "a7s") — NOT the graph's `rag_domain:<id>` name.
    pub id: String,
    /// Human-readable display name (same as `id` today).
    pub name: String,
    /// Domain description stored on the `rag_domain` node.
    pub description: String,
    /// Number of `rag_chunk` nodes currently stored for the domain.
    pub chunk_count: u64,
    /// Number of distinct source paths the chunks came from.
    pub source_count: u64,
    /// Deterministic corpus version (16-hex) from the config.
    pub corpus_version: String,
    /// Total number of tokens across all chunks in the domain.
    pub token_count: u64,
    /// Number of `rag_entity` nodes stored for the domain.
    pub entity_count: u64,
    /// Number of `rag_relationship` edges stored for the domain.
    pub relationship_count: u64,
}

/// A single retrieved chunk.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RagChunkResult {
    pub domain: String,
    pub source_path: String,
    pub chunk_index: u32,
    pub text: String,
    pub score: f32,
}

/// Sized service wrapper for the shared embedder.
///
/// `std::any::Any` cannot hold unsized `Arc<dyn Embedder>` directly (its
/// `downcast` requires `T: Sized`), so the embedder is registered as this
/// sized wrapper via `ServiceRegistry::register_service("embedder", …)` and
/// resolved back to its `Arc<dyn Embedder>` on retrieval.
#[derive(Clone)]
pub struct EmbedderService(pub std::sync::Arc<dyn Embedder>);

// ============================================================================
// Messages
// ============================================================================

pub enum RagMessage {
    /// Semantic search within a domain.
    Query {
        domain: String,
        query: String,
        top_k: usize,
        reply_to: oneshot::Sender<Result<Vec<RagChunkResult>>>,
    },
    /// List all domains with their corpus state (chunk/source/entity counts,
    /// corpus version).
    ListDomains {
        reply_to: oneshot::Sender<Result<Vec<RagDomainInfo>>>,
    },
    /// Discover ingestion manifests (`ingest.yaml` per platform) — the list of
    /// "ingest scripts".
    ListManifests {
        project_root: PathBuf,
        reply_to: oneshot::Sender<Result<Vec<RagManifestInfo>>>,
    },
    /// Ingest a GraphRAG `ingest.yaml` config (the canonical format).
    /// `project_root` is optional: relative source paths resolve against the
    /// manifest's own directory when None — RAG is project-independent.
    IngestGraphConfig {
        manifest_path: PathBuf,
        project_root: Option<PathBuf>,
        reply_to: oneshot::Sender<Result<IngestReport>>,
    },
    /// Clear the manifest's target domain and then ingest from scratch — the
    /// durable "reingest" that REPLACES stale content instead of appending to
    /// it (for changed docs/sources/embedding settings).
    ReingestGraphConfig {
        manifest_path: PathBuf,
        project_root: Option<PathBuf>,
        reply_to: oneshot::Sender<Result<IngestReport>>,
    },
    /// List persisted per-source ingest status for a domain (from the
    /// KnowledgeStore's `rag_source` nodes, so it shows without re-ingesting).
    ListSources {
        domain: String,
        reply_to: oneshot::Sender<Result<Vec<rag_ingest::SourceStatus>>>,
    },
    /// Find code-interface nodes (AstFunction/AstClass) matching a query.
    FindInterfaces {
        domain: String,
        query: String,
        top_k: usize,
        reply_to: oneshot::Sender<Result<Vec<RagChunkResult>>>,
    },
    /// Set the UI-selected default domain. `None` clears it. Search tools whose
    /// `domain` is empty resolve against this value inside the actor.
    SetDefaultDomain {
        domain: Option<String>,
    },
}

// ============================================================================
// RagActor
// ============================================================================

pub struct RagActor {
    /// Data plane: the user-level KnowledgeStore (`~/.spire/knowledge`).
    /// All `rag_domain` / `rag_chunk` / `rag_entity` nodes live here.
    knowledge_tx: mpsc::Sender<MemoryGraphMessage>,
    /// Provenance: the project graph. Kept for the
    /// `Project → uses_platform_rag(platform_id, corpus_version)` edge.
    memory_graph_tx: mpsc::Sender<MemoryGraphMessage>,
    embedder: std::sync::Arc<dyn Embedder>,
    /// UI-selected default domain (`rag/set-domain`). Actor-owned so search
    /// tools with an empty `domain` resolve against a single, mailbox-serialized
    /// value instead of a shared `Arc<Mutex<Option<String>>>`.
    default_domain: Option<String>,
}

impl RagActor {
    pub fn new(
        knowledge_tx: mpsc::Sender<MemoryGraphMessage>,
        memory_graph_tx: mpsc::Sender<MemoryGraphMessage>,
        embedder: std::sync::Arc<dyn Embedder>,
    ) -> Self {
        Self {
            knowledge_tx,
            memory_graph_tx,
            embedder,
            default_domain: None,
        }
    }

    /// Legacy convenience: single graph for both data plane and provenance.
    /// Used by the standalone binary and tests that don't split stores.
    pub fn new_shared(
        memory_graph_tx: mpsc::Sender<MemoryGraphMessage>,
        embedder: std::sync::Arc<dyn Embedder>,
    ) -> Self {
        Self {
            knowledge_tx: memory_graph_tx.clone(),
            memory_graph_tx,
            embedder,
            default_domain: None,
        }
    }

    /// Construct with the embedder resolved from the shared service registry.
    /// Falls back to the no-op embedder (degraded mode) if not registered.
    pub fn from_registry(
        knowledge_tx: mpsc::Sender<MemoryGraphMessage>,
        memory_graph_tx: mpsc::Sender<MemoryGraphMessage>,
        registry: std::sync::Arc<ServiceRegistry>,
    ) -> Self {
        let embedder = registry
            .get_service::<EmbedderService>("embedder")
            .map(|s| s.0.clone())
            .unwrap_or_else(|| std::sync::Arc::new(NoopEmbedder) as std::sync::Arc<dyn Embedder>);
        Self {
            knowledge_tx,
            memory_graph_tx,
            embedder,
            default_domain: None,
        }
    }

    /// Resolve a caller-supplied domain against the actor's default. An empty
    /// `domain` means "use the UI-selected default" (falling back to "" — the
    /// pre-actor behaviour when no default was ever set).
    fn resolve_domain(&self, domain: String) -> String {
        if domain.is_empty() {
            self.default_domain.clone().unwrap_or_default()
        } else {
            domain
        }
    }

    /// Semantic query over a domain's `rag_chunk` nodes, cosine-first.
    async fn query(&self, domain: &str, query: &str, top_k: usize) -> Result<Vec<RagChunkResult>> {
        semantic_retrieve(
            &self.knowledge_tx,
            &self.embedder,
            domain,
            query,
            None,
            top_k,
        )
        .await
    }

    /// Count graph nodes/edges of a subtype, filtered by a domain property.
    async fn count_type(
        &self,
        store: &mpsc::Sender<MemoryGraphMessage>,
        subtype: &str,
        domain: &str,
    ) -> Result<u64> {
        let (tx, rx) = oneshot::channel();
        store
            .send(MemoryGraphMessage::QueryAttrNodes {
                node_type: Some("Unknown".to_string()),
                subtype: Some(subtype.to_string()),
                name: None,
                limit: Some(1_000_000),
                reply_to: tx,
            })
            .await?;
        let nodes = rx.await??;
        Ok(nodes
            .iter()
            .filter(|a| a.get("domain").and_then(|v| v.as_str()) == Some(domain))
            .count() as u64)
    }

    /// Per-domain corpus state: chunk/source/token counts + entity and
    /// relationship counts, plus the corpus version recorded as provenance.
    async fn domain_stats(&self, domain: &str) -> Result<RagDomainInfo> {
        let (tx, rx) = oneshot::channel();
        self.knowledge_tx
            .send(MemoryGraphMessage::QueryAttrNodes {
                node_type: Some("Unknown".to_string()),
                subtype: Some("rag_chunk".to_string()),
                name: None,
                limit: Some(1_000_000),
                reply_to: tx,
            })
            .await?;
        let nodes = rx.await??;

        let mut chunk_count = 0u64;
        let mut sources = std::collections::BTreeSet::new();
        let mut token_count = 0u64;
        for n in &nodes {
            if n.get("domain").and_then(|v| v.as_str()) != Some(domain) {
                continue;
            }
            chunk_count += 1;
            if let Some(sp) = n.get("source_path").and_then(|v| v.as_str()) {
                sources.insert(sp.to_string());
            }
            if let Some(t) = n.get("tokens").and_then(|v| v.as_u64()) {
                token_count += t;
            }
        }

        let corpus_version = self.provenance_version(domain).await.unwrap_or_default();
        let entity_count = self
            .count_type(&self.knowledge_tx, "rag_entity", domain)
            .await
            .unwrap_or(0);
        // Relationship counts are stored as edge properties; approximate by
        // counting edges via node queries is not possible — keep at 0 until
        // Phase 2 adds edge enumeration. The domain reports chunk/entity data.
        Ok(RagDomainInfo {
            id: domain.to_string(),
            name: domain.to_string(),
            description: String::new(),
            chunk_count,
            source_count: sources.len() as u64,
            corpus_version,
            token_count,
            entity_count,
            relationship_count: 0,
        })
    }

    /// Look up the `rag_provenance` node matching a domain and return its
    /// `corpus_version` (empty when none recorded).
    async fn provenance_version(&self, domain: &str) -> Option<String> {
        let (tx, rx) = oneshot::channel();
        let _ = self
            .memory_graph_tx
            .send(MemoryGraphMessage::QueryAttrNodes {
                node_type: Some("Unknown".to_string()),
                subtype: Some("rag_provenance".to_string()),
                name: None,
                limit: Some(1000),
                reply_to: tx,
            })
            .await
            .ok()?;
        let nodes = rx.await.ok()?.ok()?;
        nodes.iter().find_map(|n| {
            if n.get("domain").and_then(|v| v.as_str()) == Some(domain) {
                n.get("corpus_version")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
    }

    /// All domain summaries with descriptions, deterministically ordered.
    async fn list_domains(&self) -> Result<Vec<RagDomainInfo>> {
        let (tx, rx) = oneshot::channel();
        self.knowledge_tx
            .send(MemoryGraphMessage::QueryAttrNodes {
                node_type: Some("Unknown".to_string()),
                subtype: Some("rag_domain".to_string()),
                name: None,
                limit: Some(1000),
                reply_to: tx,
            })
            .await?;
        let nodes = rx.await??;

        let mut out = Vec::new();
        for n in nodes {
            let name = n.name().to_string();
            let id = name
                .strip_prefix("rag_domain:")
                .unwrap_or(&name)
                .to_string();
            let description = n
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let mut info = self.domain_stats(&id).await?;
            info.description = description;
            out.push(info);
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    /// Read persisted per-source status for a domain from the KnowledgeStore.
    async fn list_sources(&self, domain: &str) -> Result<Vec<rag_ingest::SourceStatus>> {
        let (tx, rx) = oneshot::channel();
        self.knowledge_tx
            .send(MemoryGraphMessage::QueryAttrNodes {
                node_type: Some("Unknown".to_string()),
                subtype: Some("rag_source".to_string()),
                name: None,
                limit: Some(1000),
                reply_to: tx,
            })
            .await?;
        let nodes = rx.await??;
        let mut out: Vec<rag_ingest::SourceStatus> = Vec::new();
        for n in nodes {
            if n.get("domain").and_then(|v| v.as_str()) != Some(domain) {
                continue;
            }
            out.push(rag_ingest::SourceStatus {
                id: n
                    .get("source_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                source_type: n
                    .get("source_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                status: n
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                reason: n
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                chunks: n.get("chunks").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                files: n.get("files").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            });
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    /// Ingest a canonical `ingest.yaml` via the GraphRAG pipeline.
    async fn ingest_graph_config(
        &self,
        manifest_path: &Path,
        project_root: Option<&Path>,
    ) -> Result<IngestReport> {
        let ctx = rag_ingest::IngestContext {
            knowledge_tx: self.knowledge_tx.clone(),
            memory_graph_tx: self.memory_graph_tx.clone(),
            embedder: self.embedder.clone(),
        };
        rag_ingest::ingest_graph_config(&ctx, manifest_path, project_root).await
    }

    /// Clear the manifest's target domain (chunks/entities/source status) and
    /// then ingest from scratch, so changed content replaces — not appends to —
    /// the existing corpus. The `rag_domain` node survives (re-merged on ingest).
    async fn reingest_graph_config(
        &self,
        manifest_path: &Path,
        project_root: Option<&Path>,
    ) -> Result<IngestReport> {
        let domain = rag_ingest::resolve_domain_for_manifest(manifest_path)?;
        let cleared = rag_ingest::clear_domain(&self.knowledge_tx, &domain).await?;
        info!(
            "rag: reingest '{domain}': cleared {} chunk(s), {} entit(y/ies), {} source(s)",
            cleared.chunks, cleared.entities, cleared.sources
        );
        self.ingest_graph_config(manifest_path, project_root).await
    }
}

/// Shared semantic retrieval over `rag_chunk` nodes.
///
/// Re-embeds the candidate chunk texts with the shared embedder and scores
/// each with **cosine similarity**. When the embedder returns zero vectors
/// (the no-op fallback), scores fall back to lexical Jaccard.
async fn semantic_retrieve(
    memory_graph_tx: &mpsc::Sender<MemoryGraphMessage>,
    embedder: &std::sync::Arc<dyn Embedder>,
    domain: &str,
    query: &str,
    source_type_filter: Option<&str>,
    top_k: usize,
) -> Result<Vec<RagChunkResult>> {
    let q_emb = embedder.embed(query).await?;

    let (tx, rx) = oneshot::channel();
    memory_graph_tx
        .send(MemoryGraphMessage::QueryAttrNodes {
            node_type: Some("Unknown".to_string()),
            subtype: Some("rag_chunk".to_string()),
            name: None,
            limit: Some(500),
            reply_to: tx,
        })
        .await?;
    let nodes = rx.await??;

    let mut texts: Vec<String> = Vec::new();
    let mut meta: Vec<(String, u32)> = Vec::new(); // (source_path, chunk_index)
    for n in &nodes {
        if n.get("domain").and_then(|v| v.as_str()) == Some(domain) {
            if let Some(st) = source_type_filter {
                if n.get("source_type").and_then(|v| v.as_str()) != Some(st) {
                    continue;
                }
            }
            if let (Some(text), Some(source_path), Some(chunk_index)) = (
                n.get("text").and_then(|v| v.as_str()),
                n.get("source_path").and_then(|v| v.as_str()),
                n.get("chunk_index").and_then(|v| v.as_u64()),
            ) {
                texts.push(text.to_string());
                meta.push((source_path.to_string(), chunk_index as u32));
            }
        }
    }
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let emb_batch = embedder.embed_batch(&texts).await.unwrap_or_default();
    let noop = emb_batch
        .iter()
        .flat_map(|e| e.vector.iter())
        .all(|x| *x == 0.0);

    let mut scored: Vec<RagChunkResult> = Vec::new();
    for (i, (source_path, chunk_index)) in meta.iter().enumerate() {
        let text = &texts[i];
        let score = if !noop {
            emb_batch
                .get(i)
                .and_then(|e| cosine(&q_emb.vector, &e.vector))
                .unwrap_or(0.0)
        } else {
            jaccard(query, text)
        };
        if score > 0.0 {
            scored.push(RagChunkResult {
                domain: domain.to_string(),
                source_path: source_path.clone(),
                chunk_index: *chunk_index,
                text: text.clone(),
                score,
            });
        }
    }
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(top_k.max(1));
    Ok(scored)
}

/// Cosine similarity between two equal-length vectors (None on degenerate).
fn cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return None;
    }
    Some(dot / (na.sqrt() * nb.sqrt()))
}

/// Lexical Jaccard token overlap of two strings (degraded-mode fallback).
fn jaccard(a: &str, b: &str) -> f32 {
    let tokens = |s: &str| -> std::collections::HashSet<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .map(|t| t.to_lowercase())
            .filter(|t| !t.is_empty())
            .collect()
    };
    let (qa, qb) = (tokens(a), tokens(b));
    if qa.is_empty() || qb.is_empty() {
        return 0.0;
    }
    let inter = qa.intersection(&qb).count();
    let union = qa.union(&qb).count();
    if union == 0 {
        return 0.0;
    }
    inter as f32 / union as f32
}

/// Find code-interface nodes in a domain (source_type == "code"), cosine-first.
async fn find_interfaces(
    memory_graph_tx: &mpsc::Sender<MemoryGraphMessage>,
    embedder: &std::sync::Arc<dyn Embedder>,
    domain: &str,
    query: &str,
    top_k: usize,
) -> Result<Vec<RagChunkResult>> {
    semantic_retrieve(
        memory_graph_tx,
        embedder,
        domain,
        query,
        Some("code"),
        top_k,
    )
    .await
}

/// Discover the available ingestion "scripts": every
/// `~/.spire/knowledge/<platform_id>/ingest.yaml`. Each entry carries its
/// resolved platform id + deterministic corpus version so the UI can show what
/// would be ingested.
fn list_manifests(project_root: &Path) -> Result<Vec<RagManifestInfo>> {
    let _ = project_root;
    let mut out = Vec::new();

    let knowledge = knowledge_dir();
    if let Ok(entries) = std::fs::read_dir(&knowledge) {
        let mut candidates: Vec<PathBuf> = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let manifest = path.join("ingest.yaml");
            if manifest.is_file() {
                candidates.push(manifest);
            }
        }
        candidates.sort();
        for manifest in candidates {
            if let Some(info) = parse_manifest_info(&manifest) {
                out.push(info);
            }
        }
    }

    Ok(out)
}

/// Parse one canonical `ingest.yaml` into `RagManifestInfo` (skipped on parse
/// failure — the file is listed only when it is a valid GraphRAG config).
fn parse_manifest_info(path: &Path) -> Option<RagManifestInfo> {
    let yaml = std::fs::read_to_string(path).ok()?;
    let config: GraphRagConfig = serde_yaml::from_str(&yaml).ok()?;
    let manifest_dir = path
        .parent()
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let domain = rag_ingest::corpus_domain(&config, &manifest_dir);
    let corpus_version = rag_ingest::corpus_version_for(&config);
    Some(RagManifestInfo {
        domain,
        path: path.to_string_lossy().to_string(),
        corpus_version,
        description: config.pipeline.name,
    })
}

#[async_trait]
impl Actor for RagActor {
    type Message = RagMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            RagMessage::Query {
                domain,
                query,
                top_k,
                reply_to,
            } => {
                let domain = self.resolve_domain(domain);
                let r = self.query(&domain, &query, top_k).await;
                let _ = reply_to.send(r);
            }
            RagMessage::ListDomains { reply_to } => {
                let r = self.list_domains().await;
                let _ = reply_to.send(r);
            }
            RagMessage::ListManifests {
                project_root,
                reply_to,
            } => {
                let r = list_manifests(&project_root);
                let _ = reply_to.send(r);
            }
            RagMessage::IngestGraphConfig {
                manifest_path,
                project_root,
                reply_to,
            } => {
                let r = self
                    .ingest_graph_config(&manifest_path, project_root.as_deref())
                    .await;
                let _ = reply_to.send(r);
            }
            RagMessage::ReingestGraphConfig {
                manifest_path,
                project_root,
                reply_to,
            } => {
                let r = self
                    .reingest_graph_config(&manifest_path, project_root.as_deref())
                    .await;
                let _ = reply_to.send(r);
            }
            RagMessage::ListSources { domain, reply_to } => {
                let r = self.list_sources(&domain).await;
                let _ = reply_to.send(r);
            }
            RagMessage::FindInterfaces {
                domain,
                query,
                top_k,
                reply_to,
            } => {
                let domain = self.resolve_domain(domain);
                let r = find_interfaces(&self.knowledge_tx, &self.embedder, &domain, &query, top_k)
                    .await;
                let _ = reply_to.send(r);
            }
            RagMessage::SetDefaultDomain { domain } => {
                self.default_domain = domain;
            }
        }
    }
}

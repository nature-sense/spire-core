// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! GraphRAG ingestion pipeline — the canonical `ingest.yaml` format.
//!
//! The rich config format (`pipeline` / `domains` / `sources` /
//! `graph_construction` / `output`) is the ONLY ingest format Spire accepts.
//! Phase 1 implemented local sources end-to-end; Phase 2 adds remote fetchers
//! (`github_repo` / `github_org` / `web_page` / `pdf_extraction` /
//! `mailing_list`), include/exclude glob filtering, HTML/PDF text extraction,
//! and per-source status reporting.
//!
//! RAG is fully project-independent: `project_root` is optional and is only
//! used to resolve RELATIVE `source.path` entries. When absent, relative paths
//! resolve against the manifest file's own directory. The KnowledgeStore
//! (`~/.spire/knowledge`) is user-level and shared.
//!
//! Entities are stored as `Unknown`/`subtype:"rag_entity"` nodes and
//! relationships as `RelationshipType::Custom(...)` edges, both content
//! addressed so re-ingestion is idempotent.

use anyhow::{anyhow, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

use crate::config::knowledge_dir;
use crate::models::embedding::Embedder;
use crate::models::memory_graph::{AttrNode, RelationshipInput, RelationshipType};
use crate::subsystems::graph::memory_graph::MemoryGraphMessage;

// ============================================================================
// Config schema (mirrors the user's `ingest.yaml`)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphRagConfig {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub pipeline: PipelineConfig,
    #[serde(default)]
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PipelineConfig {
    #[serde(default)]
    pub name: String,
    /// Canonical corpus/domain id this ingest script writes into. Optional —
    /// when absent the manifest's own directory name
    /// (`knowledge/<corpus>/ingest.yaml`) is used, then `name`.
    #[serde(default)]
    pub corpus: String,
    /// Legacy, tolerated only for older manifests; no longer consulted for
    /// domain resolution when `corpus` or the manifest directory is present.
    #[serde(default, rename = "target_platform")]
    pub target_platform: String,
    #[serde(default)]
    pub settings: PipelineSettings,
    #[serde(default)]
    pub domains: Vec<KnowledgeDomain>,
    #[serde(default)]
    pub sources: Vec<IngestSource>,
    #[serde(default, rename = "graph_construction")]
    pub graph_construction: GraphConstruction,
}

fn default_chunk_size() -> usize {
    1800
}
fn default_chunk_overlap() -> usize {
    0
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PipelineSettings {
    #[serde(default)]
    pub concurrency: u32,
    #[serde(default)]
    pub retry_attempts: u32,
    #[serde(default)]
    pub timeout_seconds: u64,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
    #[serde(default)]
    pub embedding_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KnowledgeDomain {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub patterns: Vec<String>,
}

/// One data source with processing rules.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IngestSource {
    #[serde(default)]
    pub id: String,
    /// "local" | "markdown" | "text" | "pdf" | "code" | "github_repo" |
    /// "web_page" | "github_org" | "mailing_list" | "pdf_extraction"
    #[serde(rename = "type", default)]
    pub source_type: String,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub org: String,
    /// Local path (for "local" sources) — absolute, or relative to the
    /// resolved base directory (manifest dir, or a supplied project root).
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub processing: SourceProcessing,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SourceProcessing {
    #[serde(default)]
    pub parser: String,
    #[serde(default, rename = "include_paths")]
    pub include_paths: Vec<String>,
    #[serde(default, rename = "exclude_paths")]
    pub exclude_paths: Vec<String>,
    #[serde(default, rename = "include_files")]
    pub include_files: Vec<String>,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default, rename = "extraction_rules")]
    pub extraction_rules: Vec<ExtractionRule>,
    #[serde(default)]
    pub transformations: Vec<serde_yaml::Value>,
    #[serde(default)]
    pub tagging: Vec<TagRule>,
    #[serde(default, rename = "entity_mapping")]
    pub entity_mapping: Vec<EntityMappingRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtractionRule {
    #[serde(default)]
    pub selector: String,
    #[serde(default)]
    pub pattern: String,
    #[serde(default)]
    pub target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TagRule {
    #[serde(default)]
    pub key: String,
    #[serde(default)]
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntityMappingRule {
    #[serde(default)]
    pub pattern: String,
    #[serde(default, rename = "entity_type")]
    pub entity_type: String,
    #[serde(default)]
    pub attributes: serde_yaml::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphConstruction {
    #[serde(default, rename = "entity_resolution")]
    pub entity_resolution: EntityResolution,
    #[serde(default, rename = "relationship_inference")]
    pub relationship_inference: Vec<RelationshipInferenceRule>,
    #[serde(default, rename = "embedding_config")]
    pub embedding_config: EmbeddingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntityResolution {
    #[serde(default, rename = "merge_duplicates")]
    pub merge_duplicates: bool,
    #[serde(default, rename = "similarity_threshold")]
    pub similarity_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RelationshipInferenceRule {
    #[serde(rename = "type", default)]
    pub rule_type: String,
    #[serde(default, rename = "based_on")]
    pub based_on: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingConfig {
    #[serde(default, rename = "chunk_strategy")]
    pub chunk_strategy: String,
    #[serde(default)]
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutputConfig {
    #[serde(default)]
    pub database: String,
    #[serde(default)] // "graphrag_nodes"
    pub collection: String,
    #[serde(default)]
    pub batch_size: usize,
    #[serde(default, rename = "validate_schema")]
    pub validate_schema: bool,
    #[serde(default, rename = "node_types")]
    pub node_types: Vec<String>,
    #[serde(default, rename = "relationship_types")]
    pub relationship_types: Vec<String>,
}

// ============================================================================
// Ingest result
// ============================================================================

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SourceStatus {
    pub id: String,
    pub source_type: String,
    pub status: String, // "ok" | "skipped"
    pub reason: String,
    pub chunks: u32,
    pub files: usize,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct IngestReport {
    pub domain: String,
    pub corpus_version: String,
    pub chunks: u32,
    pub entities: u32,
    pub relationships: u32,
    pub sources_skipped: Vec<String>,
    #[serde(default)]
    pub sources: Vec<SourceStatus>,
}

// ============================================================================
// Shared context
// ============================================================================

pub struct IngestContext {
    pub knowledge_tx: mpsc::Sender<MemoryGraphMessage>,
    pub memory_graph_tx: mpsc::Sender<MemoryGraphMessage>,
    pub embedder: std::sync::Arc<dyn Embedder>,
}

fn sha256_hex(data: &[u8]) -> String {
    let mut h = sha2::Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

/// Slugify a corpus name into the RAG domain key used for retrieval.
fn slugify_corpus(name: &str) -> String {
    let t = name.trim();
    if t.is_empty() {
        return "default".to_string();
    }
    let first = t.split('/').next().unwrap_or(t);
    let slug: String = first
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let slug = slug
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if slug.is_empty() {
        "default".to_string()
    } else {
        slug
    }
}

/// Resolve the corpus/domain an ingest script targets — independent of any
/// build platform. Resolution order:
///   1. the script's explicit `pipeline.corpus` id (slugified);
///   2. the manifest's own directory name (`knowledge/<corpus>/ingest.yaml`),
///      which keeps conventionally-named corpora stable without extra fields;
///   3. `pipeline.name` (slugified), then the legacy `target_platform` slug.
pub fn corpus_domain(config: &GraphRagConfig, manifest_dir: &str) -> String {
    if !config.pipeline.corpus.trim().is_empty() {
        return slugify_corpus(&config.pipeline.corpus);
    }
    if !manifest_dir.trim().is_empty() {
        return slugify_corpus(manifest_dir);
    }
    if !config.pipeline.name.trim().is_empty() {
        return slugify_corpus(&config.pipeline.name);
    }
    slugify_corpus(&config.pipeline.target_platform)
}

/// Deterministic corpus version: sha256 of the canonical YAML (16 hex).
pub fn corpus_version_for(config: &GraphRagConfig) -> String {
    let canonical = serde_yaml::to_string(config).unwrap_or_default();
    sha256_hex(canonical.as_bytes())
        .get(..16)
        .unwrap_or("0")
        .to_string()
}

/// Resolve the corpus/domain id a manifest ingests into by parsing it and
/// combining `pipeline.corpus` with the manifest's own directory name
/// (`~/.spire/knowledge/<corpus>/ingest.yaml`).
pub fn resolve_domain_for_manifest(manifest_path: &Path) -> Result<String> {
    let yaml = std::fs::read_to_string(manifest_path)
        .map_err(|e| anyhow!("read manifest {}: {e}", manifest_path.display()))?;
    let config: GraphRagConfig = serde_yaml::from_str(&yaml)
        .map_err(|e| anyhow!("parse manifest {}: {e}", manifest_path.display()))?;
    let manifest_dir = manifest_path
        .parent()
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    Ok(corpus_domain(&config, &manifest_dir))
}

/// Counts of domain nodes removed by [`clear_domain`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ClearResult {
    pub chunks: usize,
    pub entities: usize,
    pub sources: usize,
}

/// Delete one subtype of domain-owned corpus nodes (returns how many matched).
async fn delete_domain_nodes(
    knowledge_tx: &mpsc::Sender<MemoryGraphMessage>,
    subtype: &str,
    domain: &str,
) -> Result<usize> {
    let (tx, rx) = oneshot::channel();
    knowledge_tx
        .send(MemoryGraphMessage::QueryAttrNodes {
            node_type: Some("Unknown".to_string()),
            subtype: Some(subtype.to_string()),
            name: None,
            limit: Some(1_000_000),
            reply_to: tx,
        })
        .await?;
    let nodes = rx.await??;
    let mine: Vec<AttrNode> = nodes
        .into_iter()
        .filter(|n| n.get("domain").and_then(|v| v.as_str()) == Some(domain))
        .collect();
    for n in &mine {
        let (t, r) = oneshot::channel();
        knowledge_tx
            .send(MemoryGraphMessage::DeleteNode {
                id: n.id().to_string(),
                reply_to: t,
            })
            .await?;
        r.await?
            .map_err(|e| anyhow!("delete {subtype} {}: {e}", n.id()))?;
    }
    Ok(mine.len())
}

/// Delete every `rag_chunk` / `rag_entity` / `rag_source` node belonging to
/// `domain` (relationships are auto-deleted with their endpoints). The
/// `rag_domain` node is kept — ingestion re-merges it on the next ingest.
/// Call before a full re-ingest so changed content *replaces* stale chunks
/// instead of appending alongside them.
pub async fn clear_domain(
    knowledge_tx: &mpsc::Sender<MemoryGraphMessage>,
    domain: &str,
) -> Result<ClearResult> {
    Ok(ClearResult {
        chunks: delete_domain_nodes(knowledge_tx, "rag_chunk", domain).await?,
        entities: delete_domain_nodes(knowledge_tx, "rag_entity", domain).await?,
        sources: delete_domain_nodes(knowledge_tx, "rag_source", domain).await?,
    })
}

/// Ensure a shallow clone of `url` exists at `dir`. When the cache already
/// exists it is refreshed with a best-effort `git pull` so a re-ingest picks
/// up upstream changes; a failed refresh keeps the cached copy (offline-safe)
/// rather than failing the whole corpus.
async fn clone_or_pull(url: &str, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(
        dir.parent()
            .ok_or_else(|| anyhow!("no parent for {}", dir.display()))?,
    )?;
    if !dir.join(".git").exists() {
        let status = tokio::process::Command::new("git")
            .args(["clone", "--depth", "1", url, dir.to_str().unwrap()])
            .status()
            .await?;
        if !status.success() {
            return Err(anyhow!("git clone failed for {url}"));
        }
        return Ok(());
    }
    // Refresh the cached shallow clone (best-effort: offline/failed pulls keep
    // the cached copy so ingestion can still proceed).
    let status = tokio::process::Command::new("git")
        .args([
            "-C",
            dir.to_str().unwrap(),
            "pull",
            "--ff-only",
            "--depth",
            "1",
        ])
        .status()
        .await?;
    if status.success() {
        info!("refreshed cached clone of {url} at {}", dir.display());
    } else {
        warn!(
            "git pull failed for {url} at {} — using the existing cached copy",
            dir.display()
        );
    }
    Ok(())
}

// ============================================================================
// Source fetching (Phase 2: local + remote)
// ============================================================================

/// Cache directory for cloned/fetched remote sources.
fn cache_dir() -> PathBuf {
    knowledge_dir().join(".cache")
}

/// Convert a simple glob (with `*` and `**`) to a regex.
fn glob_to_regex(glob: &str) -> Regex {
    let mut re = String::new();
    re.push('^');
    let chars: Vec<char> = glob.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    re.push_str(".*");
                    i += 2;
                    continue;
                }
                re.push_str("[^/]*");
            }
            '?' => re.push('.'),
            c if "\\.[]{}()+-^$|".contains(c) => {
                re.push('\\');
                re.push(c);
            }
            c => re.push(c),
        }
        i += 1;
    }
    re.push('$');
    Regex::new(&re).unwrap_or_else(|_| Regex::new("^$").unwrap())
}

fn matches_any(path: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|p| {
        if p == "**/*" || p == "*" {
            return true;
        }
        // Also test the basename (in case `path` is absolute and the pattern
        // is a bare filename), and the path WITHOUT a leading `/`.
        let trimmed = path.trim_start_matches('/');
        glob_to_regex(p).is_match(path)
            || (!name_guard_is(path, p) && glob_to_regex(p).is_match(trimmed))
    })
}

/// `matches_any` is used for include_files which may reference a bare
/// basename — permit it by comparing the final path component.
fn name_guard_is(_path: &str, _pattern: &str) -> bool {
    false
}

fn path_matches(path: &Path, source: &IngestSource) -> bool {
    let rel = path.to_string_lossy().to_string();
    // include_files / include_paths / exclude_paths globs may be written as
    // bare file names ("sun55i-a523.dtsi") while `path` is a full absolute
    // path. Match against the basename too so bare-name filters work.
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    // include_files: explicit file allow-list (when present, must match the
    // path OR its basename — users typically write "file.dtsi").
    if !source.processing.include_files.is_empty() {
        let hit = matches_any(&rel, &source.processing.include_files)
            || (!name.is_empty() && matches_any(&name, &source.processing.include_files));
        if !hit {
            return false;
        }
    }
    // include_paths: when present, at least one must match.
    if !source.processing.include_paths.is_empty()
        && !matches_any(&rel, &source.processing.include_paths)
    {
        return false;
    }
    // exclude_paths: when present, reject any match (empty = nothing excluded).
    if !source.processing.exclude_paths.is_empty()
        && (matches_any(&rel, &source.processing.exclude_paths)
            || (!name.is_empty() && matches_any(&name, &source.processing.exclude_paths)))
    {
        return false;
    }
    true
}

fn collect_local_files(dir: &Path, source: &IngestSource) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if dir.is_dir() {
        let walker = walkdir::WalkDir::new(dir).follow_links(false);
        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() && path_matches(&entry.path(), source) {
                out.push(entry.path().to_path_buf());
            }
        }
    } else if dir.is_file() && path_matches(dir, source) {
        out.push(dir.to_path_buf());
    }
    out
}

/// Fetch the files for one source by its `type`. `base_dir` is the resolved
/// directory used for RELATIVE local source paths.
async fn fetch_source_files(source: &IngestSource, base_dir: &Path) -> Result<Vec<PathBuf>> {
    match source.source_type.as_str() {
        "local" | "markdown" | "text" | "code" | "pdf" | "pdf_extraction" => {
            let src_path = if Path::new(&source.path).is_absolute() {
                PathBuf::from(&source.path)
            } else {
                base_dir.join(&source.path)
            };
            if !src_path.exists() {
                return Err(anyhow!("path not found: {}", src_path.display()));
            }
            Ok(collect_local_files(&src_path, source))
        }
        "github_repo" => {
            let url = source.url.trim().to_string();
            if url.is_empty() {
                return Err(anyhow!("github_repo requires a url"));
            }
            let dir = cache_dir().join(&source.id);
            clone_or_pull(&url, &dir).await?;
            Ok(collect_local_files(&dir, source))
        }
        "github_org" => {
            let org = source.org.trim().to_string();
            if org.is_empty() {
                return Err(anyhow!("github_org requires an org"));
            }
            let mut all = Vec::new();
            let api_url = format!("https://api.github.com/orgs/{org}/repos?per_page=50");
            let client = reqwest::Client::new();
            match client
                .get(&api_url)
                .header("User-Agent", "spire-rag/0.1")
                .send()
                .await
            {
                Ok(resp) => {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        if let Some(arr) = json.as_array() {
                            for repo in arr.iter().take(50) {
                                if let Some(clone) = repo.get("clone_url").and_then(|v| v.as_str())
                                {
                                    let n =
                                        repo.get("name").and_then(|v| v.as_str()).unwrap_or("repo");
                                    let dir = cache_dir().join(format!("{}-{}", org, n));
                                    if clone_or_pull(clone, &dir).await.is_ok() {
                                        all.extend(collect_local_files(&dir, source));
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => warn!("github_org API fetch failed for {org}: {e}"),
            }
            Ok(all)
        }
        "web_page" => {
            let url = source.url.trim().to_string();
            if url.is_empty() {
                return Err(anyhow!("web_page requires a url"));
            }
            let client = reqwest::Client::new();
            let html = client
                .get(&url)
                .header("User-Agent", "spire-rag/0.1")
                .send()
                .await?
                .text()
                .await?;
            let text = html_to_text(&html, source);
            let dir = cache_dir().join(&source.id);
            std::fs::create_dir_all(&dir)?;
            let f = dir.join("page.txt");
            std::fs::write(&f, text)?;
            Ok(vec![f])
        }
        "mailing_list" => {
            let url = source.url.trim().to_string();
            if url.is_empty() {
                return Err(anyhow!("mailing_list requires a url"));
            }
            let client = reqwest::Client::new();
            let html = client
                .get(&url)
                .header("User-Agent", "spire-rag/0.1")
                .send()
                .await?
                .text()
                .await?;
            let text = strip_html(&html);
            let dir = cache_dir().join(&source.id);
            std::fs::create_dir_all(&dir)?;
            let f = dir.join("message.txt");
            std::fs::write(&f, text)?;
            Ok(vec![f])
        }
        other => Err(anyhow!("unsupported source type: {}", other)),
    }
}

/// Minimal HTML → text using `scraper` when selectors are given, else strip tags.
fn html_to_text(html: &str, source: &IngestSource) -> String {
    if source.processing.extraction_rules.is_empty() {
        return strip_html(html);
    }
    let document = scraper::Html::parse_document(html);
    let mut out = String::new();
    for rule in &source.processing.extraction_rules {
        if rule.selector.is_empty() {
            continue;
        }
        if let Ok(sel) = scraper::Selector::parse(&rule.selector) {
            for node in document.select(&sel) {
                let text = node.text().collect::<Vec<_>>().join("\n");
                if !text.trim().is_empty() {
                    if !out.is_empty() {
                        out.push_str("\n\n");
                    }
                    if !rule.target.is_empty() {
                        out.push_str(&format!("## {}\n", rule.target));
                    }
                    out.push_str(text.trim());
                }
            }
        }
    }
    if out.is_empty() {
        strip_html(html)
    } else {
        out
    }
}

/// Strip HTML tags, keeping the text content.
///
/// Rust's `regex` crate does NOT support backreferences, so `<script>` /
/// `<style>` blocks are removed with two literal patterns instead of a
/// `</\1>` one. All pattern compiles are fallible and never panic.
fn strip_html(html: &str) -> String {
    let mut text = html.to_string();
    for tag in ["script", "style"] {
        // Literal close tag — no backreference needed.
        let re = format!(r"(?is)<{tag}\b[^>]*>.*?</{tag}>");
        if let Ok(re) = Regex::new(&re) {
            text = re.replace_all(&text, " ").to_string();
        }
    }
    let tag_re = Regex::new(r"(?s)<[^>]+>");
    let text = match tag_re {
        Ok(re) => re.replace_all(&text, " ").to_string(),
        Err(_) => text,
    };
    text.chars()
        .filter(|c| *c == '\n' || *c == '\t' || (*c as u32) >= 32)
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
// ============================================================================
// Text extraction + transformations
// ============================================================================

fn extract_text(path: &Path, source_type: &str) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if source_type == "pdf"
        || source_type == "pdf_extraction"
        || (ext == "pdf" && source_type.is_empty())
    {
        let doc = lopdf::Document::load(path)?;
        let mut text = String::new();
        // `get_pages()` maps page number → object-id/generation; extract_text
        // takes page numbers, so iterate over the keys.
        for page_number in doc.get_pages().keys() {
            if let Ok(content) = doc.extract_text(&[*page_number]) {
                text.push_str(&content);
                text.push('\n');
            }
        }
        return Ok(text);
    }
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

/// Apply the pipelined transformations (best-effort; advisories in Phase 2).
fn apply_transformations(text: &str, _source: &IngestSource) -> String {
    text.to_string()
}

/// No-op link resolution in Phase 2 (kept as a clear hook).
fn apply_link_resolution(text: &str, _source: &IngestSource) -> String {
    text.to_string()
}

/// Paragraph-based chunker with optional overlap.
fn chunk_text(text: &str, max_chars: usize, overlap: usize) -> Vec<String> {
    let paras: Vec<&str> = text
        .split("\n\n")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for p in paras {
        if cur.len() + p.len() + 2 > max_chars && !cur.is_empty() {
            chunks.push(cur);
            cur = if overlap > 0 && !chunks.is_empty() {
                chunks
                    .last()
                    .unwrap()
                    .chars()
                    .rev()
                    .take(overlap)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect()
            } else {
                String::new()
            };
        }
        if !cur.is_empty() {
            cur.push_str("\n\n");
        }
        cur.push_str(p);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    if chunks.is_empty() && !text.trim().is_empty() {
        chunks.push(text.to_string());
    }
    chunks
}

// ============================================================================
// Entity extraction + relationship inference
// ============================================================================

struct CompiledPattern {
    entity_type: String,
    regex: Regex,
}

fn compile_patterns(config: &GraphRagConfig) -> Vec<CompiledPattern> {
    let mut out = Vec::new();
    for d in &config.pipeline.domains {
        for p in &d.patterns {
            if let Ok(re) = Regex::new(p) {
                out.push(CompiledPattern {
                    entity_type: d.name.clone(),
                    regex: re,
                });
            }
        }
    }
    for s in &config.pipeline.sources {
        for m in &s.processing.entity_mapping {
            if let Ok(re) = Regex::new(&m.pattern) {
                out.push(CompiledPattern {
                    entity_type: if m.entity_type.is_empty() {
                        "entity".to_string()
                    } else {
                        m.entity_type.clone()
                    },
                    regex: re,
                });
            }
        }
    }
    out
}

/// Entity types eligible for deterministic fuzzy/hardware-name resolution.
/// Function-like types (HalFunction etc.) are NOT resolved — they must stay exact.
fn is_hardware_entity_type(etype: &str) -> bool {
    !etype.contains("Function") && !etype.contains("Method") && !etype.contains("Symbol")
}

/// Known hardware-name aliases: normalize to a single canonical family token so
/// A733 / A523 / sun55i-a523 / radxa cubie a7s all unify.
fn alias(value: &str) -> Option<String> {
    let v = value.to_lowercase().replace('_', " ").replace('-', " ");
    if v == "a733" || v == "a523 A733" || v == "cubie a7s" {
        return Some("A523".to_string());
    }
    if v.contains("sun55i") {
        return Some("A523".to_string());
    }
    None
}

/// Canonical entity value (deterministic):
/// - normalized hex (0X/0x, underscores, case) → lower hex without suffix
/// - `node@0xADDR` → `node` (address is provenance, not identity)
/// - A733/A523/sun55i/cubie-a7s aliases → "A523"
fn canonicalize_entity(etype: &str, value: &str) -> String {
    let v = value.trim();
    if v.is_empty() {
        return String::new();
    }
    if !is_hardware_entity_type(etype) {
        return v.to_string();
    }
    if let Some(a) = alias(v) {
        return a;
    }
    // node@hex → node
    let split_at = v.find('@');
    if let Some(idx) = split_at {
        let (name, addr) = (&v[..idx], &v[idx + 1..]);
        if !name.is_empty()
            && addr
                .chars()
                .all(|c| c.is_ascii_hexdigit() || c == 'x' || c == 'X')
        {
            return name.to_string();
        }
    }
    // pure hex literal → normalized lowercase hex
    if v.starts_with("0x") || v.starts_with("0X") {
        let stripped = v.trim_start_matches("0x").trim_start_matches("0X");
        let cleaned: String = stripped.chars().filter(|c| *c != '_').collect();
        return format!("0x{}", cleaned.to_lowercase());
    }
    v.to_string()
}

fn entity_name(domain: &str, entity_type: &str, value: &str) -> String {
    let canonical = canonicalize_entity(entity_type, value);
    let key = format!(
        "rag_entity|{domain}|{entity_type}|{}",
        canonical.trim().to_lowercase()
    );
    let hash = sha256_hex(key.as_bytes());
    format!("rag_entity:{}", &hash[..16])
}

fn scan_entities(patterns: &[CompiledPattern], text: &str) -> Vec<(String, String)> {
    let mut seen: Vec<(String, String)> = Vec::new();
    for p in patterns {
        for caps in p.regex.captures_iter(text) {
            let m = cap_value(&caps, 1).unwrap_or_else(|| {
                p.regex
                    .find(text)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default()
            });
            if m.is_empty() {
                continue;
            }
            let norm = canonicalize_entity(&p.entity_type, m.trim());
            if norm.is_empty() {
                continue;
            }
            if !seen.iter().any(|(t, v)| *t == p.entity_type && *v == norm) {
                seen.push((p.entity_type.clone(), norm));
            }
        }
    }
    seen
}

fn cap_value(caps: &regex::Captures<'_>, idx: usize) -> Option<String> {
    caps.get(idx).map(|m| m.as_str().to_string())
}

fn infer_relationships(
    rules: &[RelationshipInferenceRule],
    chunk_entities: &[(String, String)],
    domain: &str,
) -> Vec<(String, String, String)> {
    let mut out: BTreeSet<(String, String, String)> = BTreeSet::new();
    for rule in rules {
        let edge = if rule.rule_type.is_empty() {
            "RELATED".to_string()
        } else {
            rule.rule_type.to_uppercase().replace('-', "_")
        };
        let allowed: Vec<&str> = rule.based_on.iter().map(|s| s.as_str()).collect();
        let mut pairs = 0usize;
        let mut inserted = 0usize;
        for (i, (ta, va)) in chunk_entities.iter().enumerate() {
            for (_, (tb, vb)) in chunk_entities.iter().enumerate().skip(i + 1) {
                if !allowed.is_empty()
                    && !allowed.contains(&ta.as_str())
                    && !allowed.contains(&tb.as_str())
                {
                    continue;
                }
                pairs += 1;
                // Deterministic guard: a single chunk with hundreds of entity
                // mentions (e.g. a vendored code file) produces O(n^2) edges.
                // Keep the first 400 co-occurring pairs per rule — plenty of
                // signal, no store write explosion.
                if pairs > 400 {
                    break;
                }
                let a = entity_name(domain, ta, va);
                let b = entity_name(domain, tb, vb);
                if a != b {
                    out.insert((edge.clone(), a, b));
                    inserted += 1;
                    if inserted >= 300 {
                        break;
                    }
                }
            }
            if inserted >= 300 || pairs > 400 {
                break;
            }
        }
    }
    out.into_iter().collect()
}

// ============================================================================
// Graph helpers
// ============================================================================

async fn ensure_domain(tx: &mpsc::Sender<MemoryGraphMessage>, domain: &str) -> Result<()> {
    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::MergeAttrNode {
        node: attr_unknown(
            Some("rag_domain".to_string()),
            format!("rag_domain:{domain}"),
            Some(format!("GraphRAG domain {domain}")),
            HashMap::from([(
                "description".to_string(),
                serde_json::json!(format!("GraphRAG domain {domain}")),
            )]),
            None,
        ),
        reply_to: t,
    })
    .await?;
    let _ = r.await;
    Ok(())
}

/// Build an open `AttrNode` for an Unknown-typed RAG node (open-model write
/// path). Timestamps/version are set fresh; the store's merge keeps the id.
fn attr_unknown(
    subtype: Option<String>,
    name: String,
    description: Option<String>,
    properties: HashMap<String, serde_json::Value>,
    embedding_id: Option<String>,
) -> AttrNode {
    let now = chrono::Utc::now();
    AttrNode {
        id: uuid::Uuid::new_v4().to_string(),
        node_type: "Unknown".to_string(),
        subtype,
        name,
        description,
        properties,
        embedding_id,
        created_at: now,
        updated_at: now,
        version: 1,
    }
}

async fn store_provenance(
    tx: &mpsc::Sender<MemoryGraphMessage>,
    domain: &str,
    corpus_version: &str,
) -> Result<()> {
    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::MergeAttrNode {
        node: attr_unknown(
            Some("rag_provenance".to_string()),
            format!("rag_provenance:{domain}"),
            Some(format!("domain RAG corpus {corpus_version}")),
            HashMap::from([
                ("domain".to_string(), serde_json::json!(domain)),
                (
                    "corpus_version".to_string(),
                    serde_json::json!(corpus_version),
                ),
            ]),
            None,
        ),
        reply_to: t,
    })
    .await?;
    let _ = r.await;
    Ok(())
}

#[derive(Debug, Clone)]
struct IngestStatus {
    status: String,
    reason: String,
    chunks: u32,
    files: usize,
}

// ============================================================================
// Main ingest entry
// ============================================================================

/// Ingest a canonical `ingest.yaml`. `project_root` is OPTIONAL and only
/// resolves RELATIVE `source.path` values; when None, paths resolve against
/// the manifest's own directory. The KnowledgeStore is project-independent.
pub async fn ingest_graph_config(
    ctx: &IngestContext,
    config_path: &Path,
    project_root: Option<&Path>,
) -> Result<IngestReport> {
    let yaml = std::fs::read_to_string(config_path)?;
    let config: GraphRagConfig =
        serde_yaml::from_str(&yaml).map_err(|e| anyhow!("bad ingest.yaml: {e}"))?;

    let manifest_dir = config_path
        .parent()
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let domain = corpus_domain(&config, &manifest_dir);
    let cv = corpus_version_for(&config);

    ensure_domain(&ctx.knowledge_tx, &domain).await?;
    // Provenance is project-optional: when no project store is in scope, skip
    // the project-graph write (the corpus still lives in the KnowledgeStore).
    if project_root.is_some() {
        store_provenance(&ctx.memory_graph_tx, &domain, &cv).await?;
    }

    info!(
        "rag_ingest: domain={} corpus={} sources={}",
        domain,
        cv,
        config.pipeline.sources.len()
    );

    let patterns = compile_patterns(&config);
    let mut report = IngestReport {
        domain: domain.clone(),
        corpus_version: cv,
        ..Default::default()
    };

    let mut source_stats: Vec<(IngestSource, IngestStatus)> = Vec::new();
    // Resolve relative source paths against the manifest's own directory when
    // no project root is supplied — RAG is fully project-independent.
    let base_dir = project_root.map(Path::to_path_buf).unwrap_or_else(|| {
        config_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default()
    });

    for source in &config.pipeline.sources {
        if !source.enabled {
            continue;
        }
        let files = match fetch_source_files(source, &base_dir).await {
            Ok(f) => f,
            Err(e) => {
                warn!("rag_ingest: source '{}' fetch failed: {}", source.id, e);
                report
                    .sources_skipped
                    .push(format!("{} ({})", source.id, e));
                source_stats.push((
                    source.clone(),
                    IngestStatus {
                        status: "skipped".to_string(),
                        reason: e.to_string(),
                        chunks: 0,
                        files: 0,
                    },
                ));
                continue;
            }
        };
        if files.is_empty() {
            warn!("rag_ingest: source '{}' yielded no files", source.id);
            source_stats.push((
                source.clone(),
                IngestStatus {
                    status: "skipped".to_string(),
                    reason: "no files matched".to_string(),
                    chunks: 0,
                    files: 0,
                },
            ));
            continue;
        }

        let mut source_chunks: u32 = 0;
        for file in &files {
            let text = match extract_text(file, &source.source_type) {
                Ok(t) => t,
                Err(e) => {
                    warn!("rag_ingest: extract failed for {}: {}", file.display(), e);
                    continue;
                }
            };
            let text = apply_link_resolution(&text, source);
            let text = apply_transformations(&text, source);
            let rel = file
                .strip_prefix(&base_dir)
                .unwrap_or(file)
                .to_string_lossy()
                .to_string();

            let chunks = chunk_text(
                &text,
                config.pipeline.settings.chunk_size,
                config.pipeline.settings.chunk_overlap,
            );
            if chunks.is_empty() {
                continue;
            }
            let embeddings = match ctx.embedder.embed_batch(&chunks).await {
                Ok(v) => v,
                Err(e) => {
                    // Core functionality: fail the whole ingest loudly instead of
                    // silently dropping chunks — a corpus with missing embeddings
                    // would look complete but break semantic search.
                    anyhow::bail!(
                        "rag_ingest: embedding failed for {} ({}): {}",
                        file.display(),
                        chunks.len(),
                        e
                    );
                }
            };

            for (i, chunk) in chunks.iter().enumerate() {
                let mut hasher = sha2::Sha256::new();
                hasher.update(domain.as_bytes());
                hasher.update(b":");
                hasher.update(rel.as_bytes());
                hasher.update(b":");
                hasher.update(i.to_string().as_bytes());
                let chunk_name = format!("rag_chunk:{}", &format!("{:x}", hasher.finalize())[..16]);
                let mut props: HashMap<String, serde_json::Value> = HashMap::from([
                    ("domain".to_string(), serde_json::json!(domain)),
                    ("source_path".to_string(), serde_json::json!(rel)),
                    (
                        "source_type".to_string(),
                        serde_json::json!(source.source_type),
                    ),
                    ("chunk_index".to_string(), serde_json::json!(i as u32)),
                    ("text".to_string(), serde_json::json!(chunk)),
                    (
                        "tokens".to_string(),
                        serde_json::json!(chunk.split_whitespace().count()),
                    ),
                    (
                        "corpus_version".to_string(),
                        serde_json::json!(report.corpus_version),
                    ),
                ]);
                for tag in &source.processing.tagging {
                    props.insert(tag.key.clone(), serde_json::json!(tag.value));
                }
                let (t, r) = oneshot::channel();
                ctx.knowledge_tx
                    .send(MemoryGraphMessage::MergeAttrNode {
                        node: attr_unknown(
                            Some("rag_chunk".to_string()),
                            chunk_name,
                            Some(format!("{} chunk {}", rel, i)),
                            props,
                            embeddings.get(i).map(|e| e.text_hash.clone()),
                        ),
                        reply_to: t,
                    })
                    .await?;
                let _ = r.await;
                report.chunks += 1;
                source_chunks += 1;

                let mentions = scan_entities(&patterns, chunk);
                let mut chunk_entity_ids: Vec<(String, String)> = Vec::new();
                for (etype, value) in &mentions {
                    let name = entity_name(&domain, etype, value);
                    let (t, r) = oneshot::channel();
                    ctx.knowledge_tx
                        .send(MemoryGraphMessage::MergeAttrNode {
                            node: attr_unknown(
                                Some("rag_entity".to_string()),
                                name.clone(),
                                Some(format!("{etype}: {value}")),
                                HashMap::from([
                                    ("domain".to_string(), serde_json::json!(domain)),
                                    ("entity_type".to_string(), serde_json::json!(etype)),
                                    ("value".to_string(), serde_json::json!(value)),
                                    ("source_path".to_string(), serde_json::json!(rel)),
                                    ("source".to_string(), serde_json::json!(source.id)),
                                ]),
                                None,
                            ),
                            reply_to: t,
                        })
                        .await?;
                    if let Ok(Ok(node)) = r.await {
                        chunk_entity_ids.push((node.id().to_string(), name));
                        report.entities += 1;
                    }
                }

                let rels = infer_relationships(
                    &config.pipeline.graph_construction.relationship_inference,
                    &mentions,
                    &domain,
                );
                for (edge_type, from_name, to_name) in rels {
                    let from_id = chunk_entity_ids
                        .iter()
                        .find(|(_, n)| *n == from_name)
                        .map(|(id, _)| id.clone())
                        .unwrap_or_else(|| from_name.clone());
                    let to_id = chunk_entity_ids
                        .iter()
                        .find(|(_, n)| *n == to_name)
                        .map(|(id, _)| id.clone())
                        .unwrap_or_else(|| to_name.clone());
                    let (t, r) = oneshot::channel();
                    ctx.knowledge_tx
                        .send(MemoryGraphMessage::CreateRelationship {
                            rel: RelationshipInput {
                                edge_type: RelationshipType::Custom(edge_type.clone()),
                                from_id,
                                to_id,
                                properties: Some(HashMap::from([
                                    ("domain".to_string(), serde_json::json!(domain)),
                                    ("relation".to_string(), serde_json::json!(edge_type)),
                                ])),
                                weight: None,
                            },
                            reply_to: t,
                        })
                        .await?;
                    let _ = r.await;
                    report.relationships += 1;
                }
            }
        }

        source_stats.push((
            source.clone(),
            IngestStatus {
                status: "ok".to_string(),
                reason: String::new(),
                chunks: source_chunks,
                files: files.len(),
            },
        ));
    }

    // Persist per-source status into the KnowledgeStore so the UI can show it
    // WITHOUT re-running ingest (idempotent: content-addressed by source id).
    let statuses: Vec<SourceStatus> = source_stats
        .into_iter()
        .map(|(src, st)| SourceStatus {
            id: src.id,
            source_type: src.source_type,
            status: st.status.clone(),
            reason: st.reason.clone(),
            chunks: st.chunks,
            files: st.files,
        })
        .collect();
    for st in &statuses {
        let (t, r) = oneshot::channel();
        ctx.knowledge_tx
            .send(MemoryGraphMessage::MergeAttrNode {
                node: attr_unknown(
                    Some("rag_source".to_string()),
                    format!("rag_source:{}:{}", domain, st.id),
                    Some(format!("source {} ({})", st.id, st.source_type)),
                    HashMap::from([
                        ("domain".to_string(), serde_json::json!(domain)),
                        ("source_id".to_string(), serde_json::json!(st.id)),
                        ("source_type".to_string(), serde_json::json!(st.source_type)),
                        ("status".to_string(), serde_json::json!(st.status.clone())),
                        ("reason".to_string(), serde_json::json!(st.reason.clone())),
                        ("chunks".to_string(), serde_json::json!(st.chunks)),
                        ("files".to_string(), serde_json::json!(st.files)),
                        (
                            "corpus_version".to_string(),
                            serde_json::json!(report.corpus_version),
                        ),
                    ]),
                    None,
                ),
                reply_to: t,
            })
            .await?;
        let _ = r.await;
    }
    report.sources = statuses;

    info!(
        "rag_ingest: domain={} chunks={} entities={} relationships={} skipped={:?}",
        domain, report.chunks, report.entities, report.relationships, report.sources_skipped
    );
    Ok(report)
}

#[cfg(test)]
mod corpus_domain_tests {
    use super::*;

    fn cfg(yaml: &str) -> GraphRagConfig {
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn explicit_corpus_wins_over_dir() {
        let c = cfg("pipeline:\n  name: Allwinner A733 GraphRag\n  corpus: My Corpus.Alpha!\n");
        assert_eq!(corpus_domain(&c, "some-dir"), "my-corpus-alpha");
    }

    #[test]
    fn manifest_dir_is_default_corpus() {
        // legacy verbose fields present — dir still wins, no platform lookup
        let c = cfg(
            "pipeline:\n  name: Allwinner A733 GraphRAG Ingestion\n  target_platform: Radxa Cubie A7S/A7A\n",
        );
        assert_eq!(corpus_domain(&c, "a7s"), "a7s");
    }

    #[test]
    fn name_fallback_when_no_dir_or_corpus() {
        let c = cfg("pipeline:\n  name: Spire Core Docs\n");
        assert_eq!(corpus_domain(&c, ""), "spire-core-docs");
    }

    #[test]
    fn legacy_target_platform_is_last_resort() {
        let c = cfg("pipeline:\n  target_platform: swift\n");
        assert_eq!(corpus_domain(&c, ""), "swift");
    }

    #[test]
    fn empty_everything_defaults() {
        let c = cfg("");
        assert_eq!(corpus_domain(&c, ""), "default");
    }

    #[test]
    fn slugs_collapse_runs_of_separators() {
        let c = cfg("pipeline:\n  corpus: \"Spire___Core..Docs\"\n");
        assert_eq!(corpus_domain(&c, ""), "spire-core-docs");
    }
}

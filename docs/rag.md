# RAG Knowledge Store & Efficient Ingestion

<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (c) 2026 NatureSense -->

## RAG in one paragraph

RAG in Spire is **per-domain** and **user-level**, not project-scoped. A "domain"
(e.g. `a7s`) is a retrieval scope for one product/platform. Its corpus lives in
the shared KnowledgeStore (`config::knowledge_dir()`, default `~/.spire/knowledge`)
as `rag_domain`, `rag_chunk`, `rag_entity`, `rag_relationship`, and `rag_source`
nodes in a single SeleneDB instance. `RagActor::Query` re-embeds the query and
scores chunks by cosine similarity (lexical Jaccard fallback in degraded mode).
See also [`docs/spatial.md`](spatial.md) for place-aware (spatial + semantic)
retrieval over the same store.

## Overview

```
ingest.yaml (GraphRagConfig)
        │  RagActor::IngestGraphConfig
        ▼
rag_ingest::ingest_graph_config(ctx, manifest_path, project_root)
  1. resolve domain id              (pipeline.target_platform → clean id)
  2. compute corpus_version         (deterministic 16-hex fingerprint)
  3. fetch + extract sources        (local files, web pages, …)
  4. chunk text                     (chunk size / overlap)
  5. batch-embed chunks             (ctx.embedder.embed_batch — one call, not N)
  6. store                          (chunks + entities + relationships + source status)
        │
        ▼
MemoryGraphActor (knowledge store)     ←─── RagActor::Query
                                        │   embed query → vector search →
                                        │   cosine score → top_k chunks
                                        │   (Jaccard fallback when no vectors)
```

## Components

| Component | Role |
| --- | --- |
| `actors::rag::RagActor` | Retrieval entry point (`Query`, `FindInterfaces`) and ingestion control (`IngestGraphConfig`, `ReingestGraphConfig`, `ListManifests`, `ListDomains`, `ListSources`). |
| `actors::rag_ingest` | Parser + executor for `ingest.yaml` (`GraphRagConfig`); pure functions `resolve_domain` and `corpus_version_for`. |
| `subsystems::graph::MemoryGraphActor` | The store behind both the project graph and the KnowledgeStore. All RAG reads/writes go through its messages. |
| `models::embedding::Embedder` | The embedding seam. Ingest and retrieval both use the *same* shared `Arc<dyn Embedder>`. |
| `embedder::CandleEmbedder` / `NoopEmbedder` | Local 384-d embedder; fail-loud degraded-mode placeholder. |

The shared embedder is registered in the `ServiceRegistry` under `"embedder"` as
`EmbedderService(Arc<dyn Embedder>)` (a **sized** wrapper — `Any::downcast` needs
`Sized`, so a raw `Arc<dyn Embedder>` cannot be stored). `RagActor::from_registry`
resolves it and falls back to `NoopEmbedder` in degraded mode.

## The `ingest.yaml` manifest (`GraphRagConfig`)

A platform's RAG corpus is described by one YAML manifest. `rag_ingest` models it
with `GraphRagConfig`, whose sections map to the config structs
(`PipelineConfig`, `KnowledgeDomain`, `IngestSource`, `SourceProcessing`,
`ExtractionRule`, `TagRule`, `EntityMappingRule`, `GraphConstruction`,
`EntityResolution`, `RelationshipInferenceRule`, `EmbeddingConfig`,
`OutputConfig`):

```yaml
# ~/.spire/knowledge/<domain>/ingest.yaml  (conceptual shape)
pipeline:
  target_platform: a7s          # → resolve_domain → "a7s"
knowledge_domain:
  name: a7s
  description: A7S detection platform docs
sources:
  - id: docs
    type: local_files           # or web/url
    paths: ["docs/**/*.md"]
    processing:                 # chunking, extraction, tagging rules
      chunk_size: 800
      chunk_overlap: 80
graph_construction:
  entities:                     # entity-mapping rules (text → rag_entity)
  relationships:                # relationship-inference rules (→ rag_relationship)
output:
  corpus_version: auto          # otherwise computed deterministically
```

Key fields used by the engine:

- **`pipeline.target_platform`** → `resolve_domain()` produces the clean domain id
  (the retrieval scope).
- **`corpus_version_for(config)`** returns a deterministic 16-hex fingerprint of
  the config. The version is stored on the domain node, so re-ingesting an
  unchanged manifest is a cheap no-op; changing the manifest bumps the version
  and marks the previous corpus stale.

`RagActor::ListManifests` discovers these manifests (`ingest.yaml` per platform)
as the list of "ingest scripts" shown in the UI.

## The ingestion pipeline

`RagActor::IngestGraphConfig { manifest_path, project_root, reply_to }` hands off
to `rag_ingest::ingest_graph_config(ctx, manifest_path, project_root)`:

1. **Resolve the domain** — `resolve_domain(target_platform)` → clean id
   (`a7s`), the retrieval scope.
2. **Compute the corpus version** — `corpus_version_for(config)` (deterministic
   fingerprint). If the stored domain node already has this version, ingestion is
   a no-op for unchanged sources.
3. **Fetch & extract sources** — local files (`paths` + globs), and any web/URL
   sources. Extraction rules normalize the text; failed sources are recorded as
   `SourceStatus` with a reason and skipped — one bad source never aborts the
   whole domain.
4. **Chunk** — each extracted document is split by `chunk_size` / `chunk_overlap`
   (`SourceProcessing`).
5. **Embed in batches** — the *whole batch* of chunks is embedded with a single
   `embedder.embed_batch(&chunks)` call (never one `embed()` per chunk — that is
   orders of magnitude slower). Chunks are stored via
   `MemoryGraphMessage::StoreNode` / `MergeAttrNode` (with or without the
   pre-computed vector).
6. **Build the graph** — entity-mapping and relationship-inference rules turn
   chunk text into `rag_entity` nodes and `rag_relationship` edges
   (`GraphConstruction`, `EntityResolution`, `RelationshipInferenceRule`).
7. **Persist per-source status** — `rag_source` nodes record how many chunks each
   source produced so `RagActor::ListSources` reports state **without
   re-ingesting**.

The `IngestContext` carries both graph channels (the KnowledgeStore `knowledge_tx`
and the project graph `memory_graph_tx`) plus the shared embedder, so the
pipeline never blocks on service lookups.

## Retrieval

`RagActor::Query { domain, query, top_k }`:

1. Embed the query with the shared embedder.
2. Vector-search the domain's `rag_chunk` nodes (SeleneDB vector index) and score
   with **cosine similarity** on the real vectors.
3. Return the top `top_k` chunks as `RagChunkResult { domain, source_path,
   chunk_index, text, score }`.

**Degraded mode:** when the registered embedder is the no-op placeholder (zero
vectors), scoring falls back to **lexical Jaccard overlap** so the tool keeps
working — but this is a fallback, not a target. `RagActor::FindInterfaces`
queries `AstFunction`/`AstClass` interface nodes the same way.

## Versioning & idempotency

- `corpus_version_for` is a **pure function of the config** → identical manifests
  always yield the same version, so re-ingesting after a crash or a restart
  converges instead of duplicating.
- The version is stored on the `rag_domain` node; `RagDomainInfo.corpus_version`
  exposes it to the UI.
- Changed manifests produce a new version. `IngestGraphConfig` is an idempotent
  **upsert** — chunk/entity names are content-addressed, so unchanged content
  converges (no duplicates) while changed content is added under new names.
- `ReingestGraphConfig` (the RAG panel **Reingest** button /
  `rag/reingest-graph-config`) is a true **replace**: it resolves the manifest's
  domain, clears its `rag_chunk`/`rag_entity`/`rag_source` nodes
  (`rag_ingest::clear_domain`, relationships auto-delete, the `rag_domain` node
  survives), then ingests from scratch — stale chunks/entities from older
  content are pruned, not left behind. Use it after source content, the
  manifest, or the embedding config changes.
- Per-source status nodes make the pipeline **resumable**: `ListSources` shows
  exactly what landed and why anything was skipped.

## Refreshing remote sources & re-ingesting

- **GitHub sources refresh themselves.** `github_repo` and `github_org` clones
  live in `~/.spire/knowledge/.cache/<source-id>` as shallow `--depth 1` clones.
  On every ingest `rag_ingest` runs a best-effort `git pull --ff-only` when the
  cached clone already exists, so re-ingestion picks up pushed changes without
  deleting the cache. A failed pull (offline, auth, …) keeps the cached copy and
  ingests from it rather than failing the corpus.
- **Re-ingest = clear + ingest.** The durable way to refresh a corpus whose
  source changed is `RagMessage::ReingestGraphConfig` — the RAG panel's
  **Reingest** button (RPC `rag/reingest-graph-config`). It resolves the
  manifest's domain (`rag_ingest::resolve_domain_for_manifest`), removes the
  domain's corpus nodes, then ingests from scratch. Plain **Ingest** remains an
  idempotent upsert for re-running an unchanged manifest.

## Practices for efficient RAG ingestion

- **Always use `embed_batch`**, never a loop of `embed()`. Embedding dominates
  ingest cost; batching amortizes the model forward pass.
- **Keep the corpus version deterministic** — never write a random or timestamp
  fingerprint into `corpus_version`; it must be a pure function of the manifest
  for idempotent re-ingest.
- **Scope retrieval to the domain** — pass the resolved domain id everywhere;
  the chunk/entity queries filter on `rag_domain:<id>` so corpora never bleed
  into each other.
- **Fail per source, not per corpus** — record `SourceStatus` for each source and
  continue; a single 404 or parse error should not block the rest of the domain.
- **Reuse the shared embedder** — construct `CandleEmbedder` once (it does
  blocking model I/O) and share it via `EmbedderService`; do not instantiate one
  per actor or per request.
- **Prefer `StoreNodeWithEmbedding` / `MergeAttrNode` over two-step
  store-then-embed** so the vector lands atomically with the node.
- **Use the KnowledgeStore, not the project graph**, for RAG corpora — the two
  are separate SeleneDB instances with different lifetimes and cleanup policies.
- **Monitor degraded mode** — if `NoopEmbedder` is ever in play, retrieval
  silently drops to Jaccard; surface a warning when the `"embedder"` service is
  missing at startup.

# Use Cases

<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (c) 2026 NatureSense -->

Real-world scenarios this crate is built for. Each describes the actors and
messages involved.

## 1. Per-platform RAG corpus ingestion

**Goal:** build a searchable knowledge base for a product platform (e.g. `a7s`)
from docs, source files, and web pages.

1. `RagActor::ListManifests` → discover `ingest.yaml` manifests per platform.
2. `RagActor::IngestGraphConfig` → `rag_ingest::ingest_graph_config` fetches,
   extracts, chunks, batch-embeds, and stores `rag_chunk`/`rag_entity`/
   `rag_relationship`/`rag_source` nodes in the KnowledgeStore.
3. `RagActor::ListDomains` / `ListSources` → surface corpus state and per-source
   status in the UI without re-ingesting.

Key actors: `RagActor`, `rag_ingest`, `MemoryGraphActor`, the shared embedder.

## 2. Semantic code & documentation search

**Goal:** answer "how does X work?" over a domain corpus.

- `RagActor::Query { domain, query, top_k }` — cosine retrieval over
  `rag_chunk` nodes (Jaccard fallback in degraded mode).
- `RagActor::FindInterfaces` — same flow over `AstFunction`/`AstClass` interface
  nodes, powering "find the symbol that …" queries.

Key actors: `RagActor`, `MemoryGraphActor`, embedder.

## 3. Tool orchestration for the coding assistant

**Goal:** let the LLM plan and execute multi-step actions (build a file, run a
fix chain).

- `ToolRouterActor::ListTools` / `CallTool` — dispatch by name, merging static
  tools (modules, web search, VS Code extension tools) with dynamic MCP tools.
- `ToolOrchestrator::ExecuteToolChainWithContext` — execute ordered tool chains,
  resolving `{{variable}}` expressions against `BuildError` context, falling back
  to the generic router for unregistered steps.

Key actors: `ToolRouterActor`, `ToolsActor`, `ToolOrchestrator`, modules,
`McpClientActor`, `TransportActor`.

## 4. MCP server lifecycle management

**Goal:** connect an editor session to external MCP servers (filesystem, git,
build analyzers, GitHub, Postgres) over stdio or HTTP.

- `McpClientActor::LoadConfigFromGraph` → `ConnectAll` → `GetServerDetails`.
- `CallTool { server_name, tool_name, … }` routes to the right server.
- `GetBuildServers` / `GetBuildSystemInfo` → discover which servers can analyze
  which build files (build MCP servers self-describe via `_build_system`).

Key actors: `McpClientActor`, `McpClientManager`.

## 5. Chat with embedded widgets

**Goal:** LLM chat messages that render rich widgets (build lists, radio groups,
progress bars) in the extension UI.

- `ChatActor::Append` stores messages with an opaque `widget` JSON payload;
  `UpdateWidget` mutates a widget's state; `GetActive`/`GetHistory`/`Clear`/
  `SetTitle` manage dialogs.
- `ProgressActor::Publish`/`Subscribe` drives progress UI (e.g. startup phases,
  ingestion progress).

Key actors: `ChatActor`, `ProgressActor`.

## 6. LLM completions with role/model selection

**Goal:** call DeepSeek with a planning or coding model, optionally with tools.

- `LlmActor::Complete { role: Planning | Coding | Default }` selects the model.
- `CompleteWithTools` attaches OpenAI-compatible `tools` arrays (deduplicated by
  sanitized name).
- `Stream` returns an `mpsc::Receiver<String>` for token streaming.
- Config is updateable at runtime via `UpdateConfig` (or persisted globally via
  `config::set_global_llm_config_key`).

Key actors: `LlmActor`, `config`.

## 7. Cross-platform build metadata & cross-files

**Goal:** build one codebase for host + embedded targets (rpi5, rock3c).

- `platform::Platform` / `CrossSpec::for_platform` generate Cargo `.cargo/config`
  snippets and Meson cross files, consumed by `spire-code`'s build actors.
- `build_types` types (`BuildMetadata`, `DomainEditability`, …) are the
  serialization contract shared with the MCP build servers.

Key modules: `platform`, `build_types`.

## 8. Live project file watching

**Goal:** keep the extension in sync with disk changes.

- `FileWatcherActor::StartWatching` emits `InitialScan`, then debounced
  `Batch` notifications (500 ms quiet window) via the output channel.

Key actors: `FileWatcherActor`, `analyzer::scanner`.

## 9. Local project analysis

**Goal:** build a semantic file tree (language, role, line estimates) for
project understanding.

- `analyzer::tree_builder::build_file_tree(root, no_ignore)` returns a
  `DirectoryNode` tree from `scanner::scan_directory`.
- Used by `spire-code`'s `ProjectAnalyzerActor` and by `FileWatcherActor`.

Key modules: `analyzer::scanner`, `analyzer::tree_builder`.

## 10. Spatial queries over geotagged nodes

**Goal:** answer "what is near/inside/overlapping this place?" over nodes that
carry WGS84 coordinates or geometries (sensor deployments, survey zones,
monitoring sites).

1. Tag nodes before storing: `AttrNode::set_geo_point(Point::new(lng, lat))` for
   points or `AttrNode::set_spatial_geometry(&Geometry)` for polygons/lines —
   both derive the scalar bounding-box columns used by the GQL pre-filter.
2. `MemoryGraphMessage::SpatialQuery { query, node_type, subtype, limit }` with a
   `SpatialQuery` variant:
   - `BoundingBox { rect }` — nodes whose stored bounding box is inside `rect`.
   - `Radius { center, radius_meters }` — nodes within N geodesic meters.
   - `Nearest { center, k }` — the k closest nodes, distances in meters.
   - `Contains { geometry }` / `Intersects { geometry }` — arbitrary geometry
     predicates (point-in-zone, overlap, touching).
3. Combine with semantic retrieval: geotagged `rag_chunk`-style nodes can be
   pre-filtered by region and then cosine-ranked (or vice-versa) for
   place-aware RAG.

Key modules: `spatial` (functions), `AttrNode` spatial helpers, and the
`SpatialQuery` message. See [`docs/spatial.md`](spatial.md).

## 11. Vector-tile map UIs

**Goal:** an interactive map viewport that renders geotagged nodes as slippy-map
vector tiles (Singapore-scale, dynamically filtered).

1. Compute the viewport's tile coordinates with
   `spatial::tile_bounds(z, x, y)` / `point_to_tile(point, z)`.
2. Ask `TileActor::GetTile` (or `GetTileFeatures`) for each tile — feature sets
   are LRU-cached per `(filters, z, x, y)`, so panning reuses tiles.
3. `GetTile` returns MVT bytes (via `crate::tiles::encode_tile`) that the map
   renderer draws; `GetTileFeatures` returns the underlying nodes for GeoJSON
   or custom rendering.

Key actors: `TileActor`, `MemoryGraphActor`; key modules: `spatial`, `tiles`.

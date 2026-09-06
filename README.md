# spire-core

<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (c) 2026 NatureSense -->

**Spire core process** — a Rust actor system with an MCP client, local embedding,
and a SeleneDB-backed knowledge graph. It runs as a subprocess of the Spire VS
Code extension and provides the shared platform services that the `spire-code`
crate (coding/development actors) builds on.

This document is the crate overview. Detailed guides live in [`docs/`](docs/):

| Guide | Contents |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | Actor model, message-passing, subsystems, persistence |
| [`docs/api.md`](docs/api.md) | Per-module API reference |
| [`docs/rag.md`](docs/rag.md) | RAG knowledge store and efficient ingestion |
| [`docs/use-cases.md`](docs/use-cases.md) | Real-world usage patterns |
| [`docs/examples.md`](docs/examples.md) | Runnable code examples |
| [`docs/spatial.md`](docs/spatial.md) | WGS84 spatial queries over the memory graph |
| [`docs/best-practices.md`](docs/best-practices.md) | Recommended patterns |
| [`docs/anti-patterns.md`](docs/anti-patterns.md) | Things to avoid |

---

## What it does

`spire-core` is a **library**, not a binary. It packages the core services of the
Spire AI coding platform:

- **Actor framework** — every service is an actor (`Actor`, `ChildActor`, or a
  `Subsystem` group) running on Tokio, communicating via typed `mpsc` mailboxes
  and oneshot request/reply. The framework itself lives in the dependency
  `spire-actor`; `spire-core` supplies the domain actors.
- **MCP client** — connects to external MCP servers (stdio or HTTP) and
  aggregates their tools (`McpClientActor` / `McpClientManager`).
- **Knowledge graph** — `MemoryGraphActor` is the sole data store, backed by
  SeleneDB's `GraphDb` with GQL persistence, vector embeddings, WAL, and
  snapshots.
- **RAG knowledge store** — per-domain retrieval-augmented generation:
  `ingest.yaml`-driven ingestion (`rag_ingest`) and cosine-similarity retrieval
  (`RagActor`).
- **Spatial queries** — WGS84 geometry over the knowledge graph: bounding box,
  radius, k-nearest, contains, and intersects (`spatial` functions +
  `MemoryGraphMessage::SpatialQuery`).
- **Vector tiles** — `TileActor` + `tiles` encoder turn spatial features into
  cached MVT bytes for map UIs (`GetTile` / `GetTileFeatures`).
- **Embedding** — local text embeddings via Candle (`all-MiniLM-L6-v2`, 384-d).
- **LLM client** — DeepSeek-compatible completions, streaming, and
  tool/role-aware calls (`LlmActor`).
- **Tool orchestration** — a registry/dispatcher (`ToolRouterActor`,
  `ToolsActor`) plus a multi-step plan executor (`ToolOrchestrator`).
- **Platform modules** — long-lived child actors for filesystem, git, process,
  search, and terminal services.
- **Project analysis** — filesystem scanning and file-tree building with
  language/role/line annotations (`analyzer`).
- **Transport** — a JSON-RPC 2.0 TCP transport for talking to the VS Code
  extension (`TransportActor`).
- **Cross-platform build metadata** — typed build/domain/platform models shared
  with the MCP build servers (`build_types`, `platform`).

## Architecture at a glance

```
            ┌────────────────────────────────────────────────┐
            │           spire-code (consumer crate)          │
            │   ffi.rs / main.rs / coding actors             │
            └──────────────┬─────────────────────────────────┘
                           │ uses (library API)
┌──────────────────────────▼─────────────────────────────────┐
│                       spire-core                            │
│  ActorSystem  ── spawns ──►  actors / subsystems / modules  │
│  ServiceRegistry ── keyed sender + service lookup           │
│                                                             │
│   ├── subsystem::chat     ChatActor                         │
│   ├── subsystem::graph    MemoryGraphActor ──► GraphDb      │
│   ├── subsystem::llm      LlmActor                          │
│   ├── subsystem::mcp      McpClientActor ──► McpClientManager
│   ├── subsystem::tools    FileWatcherActor, ToolOrchestrator
│   ├── actors              Progress, SystemPrompt, RagActor, │
│   │                        ToolRouter, Tools, WebSearch     │
│   ├── modules             Filesystem, Git, Process, Search, │
│   │                        Terminal                         │
│   ├── transport           TransportActor (JSON-RPC over TCP)│
│   ├── embedder            CandleEmbedder / NoopEmbedder     │
│   └── analyzer            scanner, tree_builder             │
└─────────────────────────────────────────────────────────────┘
```

Every component is message-driven. Actors are spawned by the consumer
(`spire-code`) via `ActorSystem::spawn`, and their senders are registered in the
shared `ServiceRegistry` under stable string names. Dependencies are looked up by
name + type at `init` time and cached — never on the hot path.

## Module map

| Module | Purpose |
| --- | --- |
| [`actors`](src/actors/mod.rs) | Directly-spawned actors: `progress`, `prompt_handler`, `rag`, `rag_ingest`, `system_prompt`, `tool_providers`, `tools`, `web_search`, plus the `messages` module (`ToolInfo`, `ToolMessage`) and legacy re-exports. |
| [`subsystems`](src/subsystems/mod.rs) | Cohesive actor groups by domain: `chat`, `graph`, `llm`, `mcp`, `tools`. |
| [`modules`](src/modules/mod.rs) | Static child actors: filesystem, git, process, search, terminal. |
| [`models`](src/models/mod.rs) | Data models: `embedding` (`Embedder`, `Embedding`), `memory_graph` (`AttrNode`, relationships, transactions), `analysis`. |
| [`embedder`](src/embedder/mod.rs) | `CandleEmbedder` (all-MiniLM-L6-v2) and the fail-loud `NoopEmbedder`. |
| [`mcp`](src/mcp/mod.rs) | `McpClientManager` — external MCP server connections. |
| [`analyzer`](src/analyzer/mod.rs) | Filesystem scanning (`scanner`) and file-tree building (`tree_builder`). |
| [`graph`](src/graph.rs) | `GraphDb` — SeleneDB-backed graph with WAL/snapshot persistence and GQL. |
| [`config`](src/config.rs) | User-level config (`~/.spire`), LLM settings, knowledge dir. |
| [`build_types`](src/build_types.rs) | Cross-platform build metadata contract types. |
| [`platform`](src/platform.rs) | Cross-compilation platform definitions and cross-file generation. |
| [`spatial`](src/spatial.rs) | Pure WGS84 geometry: haversine distance, bounding boxes, contains/intersects predicates (backing `SpatialQuery`). |
| [`tiles`](src/tiles.rs) | MVT (Mapbox Vector Tile) encoding of graph features for map UIs. |
| [`transport`](src/transport/mod.rs) | `TransportActor` — JSON-RPC 2.0 over TCP. |

## Quick start

Add the dependency:

```toml
[dependencies]
spire-core = { path = "../spire-core" }
```

Spawn an actor, register it, and send a request/reply message:

```rust
use spire_core::actors::{Actor, ActorSystem, ChatActor, ChatMessage};
use tokio::sync::oneshot;

#[tokio::main]
async fn main() {
    let system = ActorSystem::new();

    // Spawn the chat actor and register its sender as a named service.
    let (chat_tx, _handle) = system.spawn(ChatActor::new());
    system.register("chat", chat_tx.clone()).unwrap();

    // Request/reply via a oneshot channel.
    let (reply_tx, reply_rx) = oneshot::channel();
    chat_tx
        .send(ChatMessage::GetActive { reply_to: reply_tx })
        .await
        .unwrap();
    let dialog = reply_rx.await.unwrap();
    println!("active dialog: {:?}", dialog);
}
```

A full working example that ingests a real RAG corpus is provided at
[`examples/rag_ingest_check.rs`](examples/rag_ingest_check.rs)
(`cargo run --example rag_ingest_check -p spire-core`).

## RAG in one paragraph

RAG corpora are **user-level and domain-scoped** (not project-scoped): a
platform's `ingest.yaml` (`GraphRagConfig`) describes sources; `rag_ingest`
fetches, extracts, chunks, batch-embeds, and stores chunks/entities/relationships
into the shared knowledge store; `RagActor::Query` re-embeds the query and scores
chunks by cosine similarity (lexical Jaccard fallback in degraded mode). See
[`docs/rag.md`](docs/rag.md).

## License

GPL-3.0-or-later. Copyright (c) 2026 NatureSense.

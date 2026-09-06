# Best Practices

<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (c) 2026 NatureSense -->

Patterns that keep `spire-core` actors fast, correct, and maintainable.

## Actor construction & wiring

- **Cache service senders at `init`/construction, never per message.** Look up
  `ctx.service::<M>("name")` once in a `ChildActor::init` (or in a constructor
  that receives the registry) and store the sender in a field. The registry is
  `Mutex`-backed — reading it on the hot path serializes every message.
- **Register every long-lived sender under a stable string name** in the
  `ServiceRegistry` at spawn time. Consumers resolve by name + type; a wrong type
  lookup returns `None` (fail fast, no silent `Any`-downcast panics).
- **`ActorSystem::spawn` returns `(Sender, JoinHandle)`** — keep the sender, and
  keep the handle if you need graceful shutdown or task supervision.

## Message passing

- **Use oneshot request/reply for everything that needs an answer.** Pass the
  `oneshot::Sender` inside the message (the `Responder<T>` pattern).
- **Never block the mailbox loop.** Long I/O, HTTP, model loads, CPU-bound work,
  or socket waits belong in a `tokio::spawn`ed task (see
  `TransportActor::CallExtension` and `TileActor::get_tile`, which runs MVT
  encoding on `spawn_blocking`).
- **Ignore the result of `reply_to.send(...)`** with `let _ =` — the caller may
  have timed out or dropped its receiver.
- **Prefer bounded channels.** The default mailbox is bounded (32); design
  messages to be cheap to enqueue.

## Tools & MCP

- **Let `ToolRouterActor` own dispatch; register tools, don't hardcode them.**
  Modules expose `ListTools`; `spire-code`'s composer registers static tools;
  MCP server tools are discovered dynamically. Adding a new backend is a
  registration, not a match-arm.
- **Keep the MCP client behind `McpClientActor`.** `McpClientManager` is not
  `Sync`; the actor is the single owner of connections.

## Graph & persistence

- **Go through `MemoryGraphActor` / GQL, never the low-level `SharedGraph`**
  (which is confined to `graph.rs`). This keeps ID mapping and schema
  invariants in one place.
- **Use `MergeAttrNode`/`MergeRelationship` for idempotent upserts** and the
  transaction stream (`OpenTransactionStream` + `Commit`/`Rollback`) when a set
  of writes must land atomically.
- **Batch GQL** (`MemoryGraphMessage::BatchGql`) for bulk operations instead of
  N round-trips.

## Spatial queries

- **Store location with the `AttrNode` helpers**, never by hand-writing JSON.
  `set_geo_point` / `set_spatial_geometry` keep the scalar
  `min_lng`/`min_lat`/`max_lng`/`max_lat` bounding-box columns in sync — the GQL
  pre-filter scans exactly those columns.
- **Coordinates must be scalar numbers.** Range predicates
  (`WHERE n.latitude >= ...`) only match native numeric properties; a
  JSON-encoded coordinate string is opaque to GQL.
- **Use `MemoryGraphMessage::SpatialQuery`, not hand-rolled geometry loops.**
  The actor pre-filters by bounding box and refines with the exact `geo`
  predicates in `crate::spatial` (bounding box, radius, k-nearest, contains,
  intersects — WGS84 lon/lat, distances in meters).
- **Query with a filter budget.** Pass `node_type`/`subtype` and keep `limit`
  small; `total_results`/`truncated` on the result tell you when you capped.
- **Serve tiles through `TileActor`** — it caches per `(filters, z, x, y)`, so
  panning a viewport never re-queries the store for the same tile. Keep the
  MVT encode on a blocking task, off the mailbox.

## Embeddings & RAG

- **Share one embedder instance** (`CandleEmbedder` is expensive to construct —
  it loads a model). Register it once as `EmbedderService(Arc<dyn Embedder>)`
  under `"embedder"`; `RagActor::from_registry` resolves it.
- **Always batch embeddings** (`embed_batch`), never a `for` loop of `embed()`.
- **Keep `corpus_version` a pure function of the manifest** so re-ingest is
  idempotent and crash-safe.
- **Fail per source, not per corpus.** Record `SourceStatus` for each source and
  continue; surface skipped sources in `ListSources`.
- **Treat `NoopEmbedder` as an explicit degraded mode.** Log loudly when the
  `"embedder"` service is missing so a silent drop to Jaccard retrieval can't
  go unnoticed.

## Configuration & tests

- **Use the env overrides** (`SPIRE_CONFIG_DIR`, `SPIRE_KNOWLEDGE_DIR`,
  `SPIRE_PLATFORM_DIR`) in tests/CI so you never touch the developer's real
  `~/.spire`.
- **Declare every feature used in `cfg!`/`cfg` in `Cargo.toml`** (e.g.
  `[features] strict-gql = []`) — undeclared features are a compile warning and
  silently evaluate to `false`.

## Filesystem writes

- **Prefer `apply_patch` (strict unified diff) over whole-file writes** for
  surgical edits — it is apply-or-fail and can never corrupt a file with stale
  content.

# Architecture

<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (c) 2026 NatureSense -->

This guide explains how `spire-core` is organised: the actor model, message
passing, subsystems, persistence, and the spatial / tile layers built on the
graph store. It assumes familiarity with Tokio.

## Architecture in one paragraph

`spire-core` is a library of **message-driven actors**. A generic framework
(`spire-actor`) provides `Actor`/`ChildActor`/`Subsystem` and a type-checked
`ServiceRegistry`; domain actors live here and talk only through typed `mpsc`
mailboxes with oneshot request/reply. The **sole data store** is
`MemoryGraphActor`, a GQL-fronted wrapper over SeleneDB's `GraphDb`. Read-only
consumers (RAG, tool orchestration, **spatial queries**, and **vector-tile
serving**) send it messages; nothing ever touches the low-level `SharedGraph`.

## The actor model

Every component is an **actor**: an object owning state that is mutated only from
its own mailbox loop. Actors communicate exclusively by sending **messages**;
they never share mutable state. The generic runtime is provided by the
`spire-actor` dependency, which is deliberately free of domain types.

### `Actor` (top-level)

```rust
pub trait Actor: Send + 'static {
    type Message: Send + 'static;
    async fn handle(&mut self, msg: Self::Message);
    fn spawn(self, rx: mpsc::Receiver<Self::Message>) -> JoinHandle<()> { /* default */ }
}
```

`ActorSystem::spawn(actor)` creates a bounded `mpsc::channel(32)`, runs the actor
on a Tokio task, and returns `(mpsc::Sender<A::Message>, JoinHandle<()>)`. The
actor processes messages until every sender is dropped.

### `ChildActor` (a wrapper with its own mini-system)

```rust
pub trait ChildActor: Send + 'static {
    type Message: Send + 'static;
    fn init(&mut self, ctx: &mut ChildContext);
    async fn handle(&mut self, ctx: &ChildContext, msg: Self::Message);
}
```

A `ChildActor` behaves like a normal actor to the outside world (one sender), but
internally it can spawn sub-actors via `ctx.spawn(...)` and resolve top-level
services via `ctx.service::<M>("name")`. **`init` runs exactly once**, before the
message loop — this is where you cache service senders. `spawn_child` /
`spawn_child_eager` adapt a `ChildActor` into a top-level `Actor`.

### `Subsystem` (a domain group of actors)

```rust
pub trait Subsystem: Send + 'static {
    type Handles;
    fn spawn(self, registry: &ServiceRegistry) -> Self::Handles;
    fn actor_count(&self) -> usize;
}
```

Subsystems group related actors (e.g. everything for "llm" or "tools"), spawn
them, and register their senders in the parent `ServiceRegistry`. Actors in one
subsystem only talk to others through the registry.

### `ActorSystem` and `ServiceRegistry`

- `ActorSystem::new()` / `with_registry(Arc<ServiceRegistry>)` — spawns actors
  onto the ambient Tokio runtime.
- `ActorSystem::spawn(actor)` — returns `(Sender, JoinHandle)`.
- `ActorSystem::register(name, sender)` / `registry()` — names a sender or gives
  access to the shared registry.

The `ServiceRegistry` is the dependency graph:

```rust
registry.register::<M>(name, sender)?;              // name a typed sender
let tx: mpsc::Sender<M> = registry.get::<M>(name)?; // type-checked lookup
registry.register_service(name, Arc<T>)?;           // name a shared service
let svc: Option<Arc<T>> = registry.get_service::<T>(name); // type-checked
```

Lookups are **type-checked via `TypeId`**: asking for `Sender<ChatMessage>` under
a name registered for a different message type returns `None`. The registry is
`Mutex`-backed, so lookups belong at `init`/construction time, not on the hot
path.

## Message passing: request/reply

The mailbox is `mpsc::Sender<M>`. Request/reply uses a **oneshot channel passed
inside the message**:

```rust
pub type Responder<T> = oneshot::Sender<Result<T, ActorError>>;

// A message variant carrying a reply channel:
pub enum ChatMessage {
    GetActive { reply_to: oneshot::Sender<Option<ChatDialog>> },
    Append    { message: ChatMessageData, reply_to: Responder<()> },
    // ...
}
```

Senders are cheap to clone and may outlive the actor (`send()` never blocks for
long). The actor replies with `let _ = reply_to.send(...);` — the caller's
receiver may have been dropped, so the result is deliberately ignored.

## Subsystem & module ownership

`subsystems/` groups actors by domain; each subdirectory owns its actors and
re-exports them at legacy flat paths via `actors/mod.rs`:

- **`subsystems::chat`** — `ChatActor`, `ChatMessage`, `ChatDialog`, `ChatMessageData`.
- **`subsystems::graph`** — `MemoryGraphActor`, `MemoryGraphMessage` (plus all graph models in `models::memory_graph`).
- **`subsystems::llm`** — `LlmActor`, `LlmMessage`, `LlmConfig`, `LlmModelRole`.
- **`subsystems::mcp`** — `McpClientActor`, `McpClientMessage`.
- **`subsystems::tools`** — `FileWatcherActor` + notifications, `ToolOrchestrator`.

`modules/` are static top-level `Actor`s (plain actors, not `ChildActor`s)
registered under stable keys (`"filesystem"`, `"git"`, `"process"`, `"search"`,
`"terminal"`). Each has its own message enum and tool list (`ListTools`), so the
LLM tool layer can discover module capabilities uniformly.

## Persistence: `GraphDb`

`MemoryGraphActor` is the only component that touches storage, through
`crate::graph::GraphDb` (a wrapper over SeleneDB's `SharedGraph`):

- `GraphDb::new_in_memory()`, `new_with_wal(&wal_path)`, `recover(&dir, graph_id)`
  (recover from the latest snapshot + WAL replay).
- Writes are serialized; reads are lock-free. Snapshots are written on `Sync`.
- The actor uses **GQL** (`execute_gql_query` / `execute_gql_write`) for all data
  access — the low-level `SharedGraph` API is not used outside `graph.rs`.

Data layout: all nodes use the `SpireNode` label; UUID strings and metadata are
stored as properties. Config is stored on `SpireConfig` nodes. Spatial location
is ordinary scalar properties (`latitude`/`longitude` or
`min_lng`/`min_lat`/`max_lng`/`max_lat` bounding boxes) plus an optional
`geometry` property carrying a GeoJSON-serialized geometry.

### Spatial queries (read path)

`MemoryGraphMessage::SpatialQuery` answers geometry questions without any
schema change. Nodes carry location via the `AttrNode` spatial helpers
(`set_geo_point`, `set_spatial_geometry`); queries then:

1. **Pre-filter with GQL range scans** over the scalar columns
   (`latitude`/`longitude` or the `min_*`/`max_*` bounding box) — cheap, indexed
   later by a SeleneDB typed index if the store grows large.
2. **Refine with exact WGS84 predicates** from `crate::spatial` (bounding box,
   radius, k-nearest, contains, intersects — distances in meters) using the
   `geo` crate on the survivors.

The result is `SpatialQueryResult { nodes, total_results, truncated }`. This is
the read path behind the vector-tile layer below.

### Vector tiles (read path)

Map UIs consume **slippy-map tiles** (`z/x/y`, Web Mercator). Two layers serve
them from the same store:

- **Projection** — `crate::spatial::tile_bounds` (tile → lon/lat window),
  `point_to_tile` and `lonlat_to_tile_coord` (features → tile-local pixels).
- **`TileActor`** (`src/actors/tile.rs`) — a read-only consumer of the graph.
  `GetTileFeatures` returns the `AttrNode`s intersecting a tile (LRU-cached per
  `(filters, z, x, y)`); `GetTile` encodes them to **MVT bytes** via
  `crate::tiles::encode_tile` (WGS84 → tile-local → clipped protobuf, one layer
  per `node_type`). The CPU-bound encode runs on a blocking task, off the
  mailbox.

`TileActor` mirrors `RagActor`: it holds a `memory_graph_tx` sender and never
owns graph state.

## ID mapping (UUID ↔ SeleneDB)

The external API uses **UUID `String` IDs** (for compatibility with the
TypeScript extension). SeleneDB uses compact `u64` ids (`NodeId`/`EdgeId`).
`MemoryGraphActor` maintains a bidirectional mapping between the two, stored as
properties on the nodes/edges themselves (`uuid` property).

## Tool types

`spire-core` owns the platform's tool metadata (`src/actors/messages.rs`,
originally in `spire-actor` before that crate was made domain-agnostic):

```rust
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}
pub struct ToolMessage {
    pub tool: String,
    pub args: serde_json::Value,
    pub response_tx: Responder<serde_json::Value>,
}
```

They are re-exported at `spire_core::actors::{ToolInfo, ToolMessage}` and used by
`LlmMessage::CompleteWithTools`, `ToolsActor`, `SystemPromptActor`, the modules,
and the `spire-code` tool providers.

## Concurrency model

- One Tokio task per actor (bounded mailbox, `mpsc::channel(32)`).
- **Never block the mailbox loop**: long or blocking work must be moved to a
  spawned task. `TransportActor`, for example, spawns a task to await an
  extension response so the reader task can keep delivering messages (avoids a
  re-entrancy deadlock). `TileActor` does the same for MVT encoding
  (`spawn_blocking`).
- Shared services are wrapped in `Arc` (e.g. the embedder is shared via
  `Arc<dyn Embedder>` inside the sized `EmbedderService`).
- The `ServiceRegistry` is `Arc<ServiceRegistry>` shared across child systems;
  it is a `Mutex<HashMap>` and is read at init time, not per message.

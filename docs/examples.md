# Examples

<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (c) 2026 NatureSense -->

Runnable patterns for the common APIs. Examples assume the dependency is added
and a Tokio runtime is available (`#[tokio::main]`).

## 1. Spawn an actor and use request/reply

Every actor call is a message plus an `mpsc::Sender<M>`; request/reply uses a
oneshot channel carried inside the message.

```rust
use spire_core::actors::{Actor, ActorSystem, ChatActor, ChatMessage};
use tokio::sync::oneshot;

#[tokio::main]
async fn main() {
    let system = ActorSystem::new();
    let (chat_tx, _handle) = system.spawn(ChatActor::new());

    let (reply_tx, reply_rx) = oneshot::channel();
    chat_tx
        .send(ChatMessage::GetActive { reply_to: reply_tx })
        .await
        .unwrap();
    println!("active dialog: {:?}", reply_rx.await.unwrap());
}
```

## 2. Register and look up services (type-checked)

```rust
use spire_core::actors::{ActorSystem, ChatActor, ChatMessage};
use spire_actor::ServiceRegistry;

let system = ActorSystem::new();
let (chat_tx, _h) = system.spawn(ChatActor::new());

// Name the sender as a service.
system.register("chat", chat_tx.clone()).unwrap();

// Look it back up — type-checked via TypeId (wrong type ⇒ None).
let tx: tokio::sync::mpsc::Sender<ChatMessage> =
    system.registry().get("chat").expect("chat service");
assert!(tx.is_same_channel(&chat_tx));
```

## 3. Write a custom actor

```rust
use async_trait::async_trait;
use spire_core::actors::{Actor, ActorError};
use tokio::sync::oneshot;

pub enum PingMsg {
    Ping { reply_to: oneshot::Sender<Result<String, ActorError>> },
}
pub struct PingActor { count: u32 }

#[async_trait]
impl Actor for PingActor {
    type Message = PingMsg;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            PingMsg::Ping { reply_to } => {
                self.count += 1;
                let _ = reply_to.send(Ok(format!("pong #{}", self.count)));
            }
        }
    }
}
```

## 4. ChildActor with a cached service sender

Use `ChildActor` when an actor needs its own mini-system or must resolve services
once at startup. Cache senders in `init` — never call `registry.get` per message.

```rust
use async_trait::async_trait;
use spire_actor::{
    spawn_child, ChildActor, ChildContext, ServiceRegistry,
};
use spire_core::actors::{ActorSystem, ChatActor, ChatMessage};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

pub enum ProbeMsg {
    GetActive(oneshot::Sender<Option<spire_core::subsystems::chat::chat::ChatDialog>>),
}
pub struct ChatProbe { chat_tx: Option<mpsc::Sender<ChatMessage>> }

#[async_trait]
impl ChildActor for ChatProbe {
    type Message = ProbeMsg;

    fn init(&mut self, ctx: &mut ChildContext) {
        self.chat_tx = ctx.service("chat");   // cached once
    }

    async fn handle(&mut self, _ctx: &ChildContext, msg: Self::Message) {
        let ProbeMsg::GetActive(reply) = msg;
        let (tx, rx) = oneshot::channel();
        let _ = self.chat_tx.as_ref().unwrap().send(ChatMessage::GetActive { reply_to: tx }).await;
        let _ = reply.send(rx.await.unwrap());
    }
}

// Spawn: register the chat actor first, then the probe.
#[tokio::main]
async fn main() {
    let system = ActorSystem::new();
    let registry: Arc<ServiceRegistry> = system.registry().clone();
    let (chat_tx, _h) = system.spawn(ChatActor::new());
    registry.register("chat", chat_tx).unwrap();

    let probe_tx = spawn_child(&system, registry, |_ctx| ChatProbe { chat_tx: None });
    let (tx, rx) = oneshot::channel();
    probe_tx.send(ProbeMsg::GetActive(tx)).await.unwrap();
    println!("active: {:?}", rx.await.unwrap());
}
```

## 5. Build a project file tree

```rust
use std::path::Path;
use spire_core::analyzer::tree_builder::build_file_tree;

let tree = build_file_tree(Path::new("."), /*no_ignore=*/ false);
println!(
    "{} files, ~{} lines ({} top-level dirs)",
    tree.total_file_count, tree.total_lines, tree.directories.len()
);
```

## 6. Store and query a graph node

```rust
use spire_core::actors::{Actor, ActorSystem, MemoryGraphActor, MemoryGraphMessage};
use spire_core::models::memory_graph::AttrNode;
use tokio::sync::oneshot;

let system = ActorSystem::new();
let (mg_tx, _h) = system.spawn(MemoryGraphActor::new());

// 1. Initialize the store against a real data directory.
let dir = tempfile::tempdir().unwrap();
let (t, r) = oneshot::channel();
mg_tx
    .send(MemoryGraphMessage::Initialize { data_dir: dir.path().to_path_buf(), reply_to: t })
    .await.unwrap();
r.await.unwrap().unwrap();

// 2. Store an AttrNode.
let node = AttrNode {
    id: "n1".to_string(),
    node_type: "example".to_string(),
    subtype: None,
    name: "hello".to_string(),
    description: None,
    properties: Default::default(),
    embedding_id: None,
    created_at: chrono::Utc::now(),
    updated_at: chrono::Utc::now(),
    version: 1,
};
let (t, r) = oneshot::channel();
mg_tx
    .send(MemoryGraphMessage::StoreAttrNode { node, reply_to: t })
    .await.unwrap();
let stored = r.await.unwrap().unwrap();
println!("stored {} (type={})", stored.name, stored.node_type);

// 3. Query it back.
let (t, r) = oneshot::channel();
mg_tx
    .send(MemoryGraphMessage::QueryAttrNodes {
        node_type: Some("example".to_string()),
        subtype: None,
        name: None,
        limit: Some(10),
        reply_to: t,
    })
    .await.unwrap();
println!("found {} nodes", r.await.unwrap().unwrap().len());
```

## 7. Spatial queries on the memory graph

```rust
// Continue from example 6's `mg_tx`. Nodes carry WGS84 location via AttrNode
// helpers; axis order is `geo::Point::new(longitude, latitude)`.
use spire_core::models::memory_graph::{AttrNode, SpatialQuery};
use tokio::sync::oneshot;

fn node(id: &str, name: &str, node_type: &str) -> AttrNode {
    AttrNode {
        id: id.to_string(),
        node_type: node_type.to_string(),
        subtype: None,
        name: name.to_string(),
        description: None,
        properties: Default::default(),
        embedding_id: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        version: 1,
    }
}

// A point sensor …
let mut sensor = node("s1", "sensor-nyc", "Sensor");
sensor.set_geo_point(geo::Point::new(-74.006, 40.7128)); // (lng, lat)

// … and a polygon "zone". set_spatial_geometry derives the bounding-box
// columns that make GQL pre-filtering work.
let mut zone = node("z1", "zone-nyc", "Zone");
let ring: geo::LineString<f64> = vec![
    (-74.5, 40.5), (-73.5, 40.5), (-73.5, 41.0), (-74.5, 41.0), (-74.5, 40.5),
].into();
zone.set_spatial_geometry(&geo::Geometry::Polygon(geo::Polygon::new(ring, Vec::new())));

for n in [sensor, zone] {
    let (t, r) = oneshot::channel();
    mg_tx.send(MemoryGraphMessage::StoreAttrNode { node: n, reply_to: t }).await.unwrap();
    r.await.unwrap().unwrap();
}

// Nodes inside a bounding box.
let (t, r) = oneshot::channel();
mg_tx
    .send(MemoryGraphMessage::SpatialQuery {
        query: SpatialQuery::BoundingBox {
            rect: geo::Rect::new(
                geo::Coord { x: -74.5, y: 40.5 },
                geo::Coord { x: -73.5, y: 41.0 },
            ),
        },
        node_type: None,
        subtype: None,
        limit: Some(50),
        reply_to: t,
    })
    .await.unwrap();
let hits = r.await.unwrap().unwrap();
println!("{} nodes in the box", hits.nodes.len());

// Everything within 100 km of the sensor (geodesic meters).
let (t, r) = oneshot::channel();
mg_tx
    .send(MemoryGraphMessage::SpatialQuery {
        query: SpatialQuery::Radius {
            center: geo::Point::new(-74.006, 40.7128),
            radius_meters: 100_000.0,
        },
        node_type: None,
        subtype: None,
        limit: None,
        reply_to: t,
    })
    .await.unwrap();
let hits = r.await.unwrap().unwrap();
for hit in hits.nodes {
    println!("{} within range at {:.0} m", hit.node.name, hit.distance_meters.unwrap_or(0.0));
}

// k nearest nodes, ordered by geodesic distance (meters).
let (t, r) = oneshot::channel();
mg_tx
    .send(MemoryGraphMessage::SpatialQuery {
        query: SpatialQuery::Nearest {
            center: geo::Point::new(-74.006, 40.7128),
            k: 3,
        },
        node_type: None,
        subtype: None,
        limit: None,
        reply_to: t,
    })
    .await.unwrap();
let hits = r.await.unwrap().unwrap();
for hit in hits.nodes {
    println!("{} — {:.0} m", hit.node.name, hit.distance_meters.unwrap_or(0.0));
}

// Features whose polygon geometry contains a point of interest.
let (t, r) = oneshot::channel();
mg_tx
    .send(MemoryGraphMessage::SpatialQuery {
        query: SpatialQuery::Contains {
            geometry: geo::Geometry::Point(geo::Point::new(-73.99, 40.75)),
        },
        node_type: Some("Zone".to_string()),
        subtype: None,
        limit: None,
        reply_to: t,
    })
    .await.unwrap();
let hits = r.await.unwrap().unwrap();
println!("{} zone(s) contain the point", hits.nodes.len());

// Features overlapping a search polygon.
let search: geo::LineString<f64> = vec![
    (-74.0, 40.6), (-73.8, 40.6), (-73.8, 40.9), (-74.0, 40.9), (-74.0, 40.6),
].into();
let (t, r) = oneshot::channel();
mg_tx
    .send(MemoryGraphMessage::SpatialQuery {
        query: SpatialQuery::Intersects {
            geometry: geo::Geometry::Polygon(geo::Polygon::new(search, Vec::new())),
        },
        node_type: None,
        subtype: None,
        limit: None,
        reply_to: t,
    })
    .await.unwrap();
let hits = r.await.unwrap().unwrap();
println!("{} feature(s) overlap the search area", hits.nodes.len());
```

See [`docs/spatial.md`](spatial.md) for the storage model, semantics, and
spatial + RAG composition patterns.

## 8. Query a RAG corpus

```rust
use std::sync::Arc;
use spire_core::actors::{RagActor, RagMessage};
use spire_core::embedder::NoopEmbedder;
use spire_core::models::embedding::Embedder;

// `mg_tx` from example 6. Shared store: one graph for data plane + provenance.
let (rag_tx, _h) = system.spawn(RagActor::new_shared(
    mg_tx.clone(),
    Arc::new(NoopEmbedder) as Arc<dyn Embedder>,
));

let (t, r) = oneshot::channel();
rag_tx
    .send(RagMessage::Query {
        domain: "a7s".to_string(),
        query: "how does object detection work".to_string(),
        top_k: 5,
        reply_to: t,
    })
    .await.unwrap();
match r.await.unwrap() {
    Ok(chunks) => println!("{} chunks", chunks.len()),
    Err(e) => eprintln!("retrieval failed: {e}"),
}
```

## 9. Call a platform module

Modules are plain `Actor`s (they own no sub-actors) — spawn them directly with
`ActorSystem::spawn` and keep the sender.

```rust
use std::path::PathBuf;
use spire_core::actors::{ActorSystem, FilesystemMessage, FilesystemModule};
use tokio::sync::oneshot;

let system = ActorSystem::new();
let (fs_tx, _h) = system.spawn(FilesystemModule::new());

let (t, r) = oneshot::channel();
fs_tx
    .send(FilesystemMessage::ReadFile {
        path: PathBuf::from("Cargo.toml"),
        reply_to: t,
    })
    .await.unwrap();
match r.await.unwrap() {
    Ok(content) => println!("Cargo.toml is {} bytes", content.len()),
    Err(e) => eprintln!("read failed: {e}"),
}
```

## 10. Global config

```rust
use spire_core::config;

println!("config dir : {}", config::config_dir().display());
println!("knowledge  : {}", config::knowledge_dir().display());

// Persist an LLM key (atomic temp-file + rename write).
let _ = config::set_global_llm_config_key("deepseek.api_key", "sk-...");
let key = config::get_global_llm_config_key("deepseek.api_key");
println!("deepseek key set: {}", key.is_some());
```

A full end-to-end ingestion example that exercises the real KnowledgeStore is in
[`examples/rag_ingest_check.rs`](../examples/rag_ingest_check.rs).
## 11. Vector tiles for the map UI

```rust
// Spawn a graph + TileActor, then request an MVT tile around Singapore.
use spire_core::actors::{
    Actor, ActorSystem, MemoryGraphActor, MemoryGraphMessage, TileActor, TileFilters,
    TileMessage,
};
use tokio::sync::oneshot;

let system = ActorSystem::new();
let (mg_tx, _h) = system.spawn(MemoryGraphActor::new());
// ... Initialize the graph, then store Sensor nodes with
//     node.set_geo_point(geo::Point::new(lng, lat)) (examples 6 & 7).

let (tile_tx, _th) = system.spawn(TileActor::new(mg_tx.clone()));

// MVT bytes for the z10 tile (x=807, y=508) containing (103.85, 1.35).
let (t, r) = oneshot::channel();
tile_tx
    .send(TileMessage::GetTile {
        filters: TileFilters {
            node_type: Some("Sensor".to_string()),
            ..Default::default()
        },
        z: 10,
        x: 807,
        y: 508,
        reply_to: t,
    })
    .await
    .unwrap();
let mvt: Vec<u8> = r.await.unwrap().unwrap();
println!("encoded {} bytes of MVT", mvt.len());

// GetTileFeatures returns the underlying nodes (e.g. for GeoJSON).
let (t, r) = oneshot::channel();
tile_tx
    .send(TileMessage::GetTileFeatures {
        filters: TileFilters::default(),
        z: 10,
        x: 807,
        y: 508,
        reply_to: t,
    })
    .await
    .unwrap();
println!("{} feature(s) in tile", r.await.unwrap().unwrap().len());
```

See [`docs/spatial.md`](spatial.md) for the tile pipeline and semantics.

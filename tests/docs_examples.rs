// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Tests that mirror the runnable examples in `docs/examples.md`, keeping the
//! documented public API compiling and exercising the documented behaviour.
//!
//! If a public API changes, update `docs/examples.md` AND this file together so
//! the two can never drift apart.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use spire_actor::{spawn_child, ChildActor, ChildContext, ServiceRegistry};
use spire_core::actors::{
    Actor, ActorSystem, ChatActor, ChatMessage, FilesystemMessage, FilesystemModule,
    MemoryGraphActor, MemoryGraphMessage, RagActor, RagMessage, TileActor, TileFilters,
    TileMessage,
};
use spire_core::analyzer::tree_builder::build_file_tree;
use spire_core::embedder::NoopEmbedder;
use spire_core::models::embedding::Embedder;
use spire_core::models::memory_graph::{AttrNode, SpatialQuery};
use spire_core::subsystems::chat::chat::ChatDialog;

use tokio::sync::{mpsc, oneshot};

/// Serializes the config example, which mutates the process-wide
/// `SPIRE_CONFIG_DIR` env var (Cargo runs integration tests in parallel in one
/// process, so the env var must not be clobbered mid-test).
static CONFIG_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ============================================================================
// docs/examples.md → §1 Spawn an actor and use request/reply
// ============================================================================

#[tokio::test]
async fn doc_example_1_spawn_and_request_reply() {
    let system = ActorSystem::new();
    let (chat_tx, _handle) = system.spawn(ChatActor::new());

    let (reply_tx, reply_rx) = oneshot::channel();
    chat_tx
        .send(ChatMessage::GetActive { reply_to: reply_tx })
        .await
        .unwrap();
    let dialog = reply_rx.await.unwrap();
    // ChatActor::new() starts with an active "default" dialog.
    let dialog = dialog.expect("a fresh ChatActor has an active default dialog");
    assert_eq!(dialog.title, "New Chat");
    assert!(dialog.messages.is_empty());
}

// ============================================================================
// docs/examples.md → §2 Register and look up services (type-checked)
// ============================================================================

#[tokio::test]
async fn doc_example_2_service_registry() {
    let system = ActorSystem::new();
    let (chat_tx, _handle) = system.spawn(ChatActor::new());

    // Name the sender as a service.
    system.register("chat", chat_tx.clone()).unwrap();

    // Look it back up — type-checked via TypeId (wrong type ⇒ None).
    let tx: mpsc::Sender<ChatMessage> = system.registry().get("chat").expect("chat service");
    assert!(!tx.is_closed(), "looked-up sender must be a live channel");
    // The looked-up sender is the same live channel: a message reaches the actor.
    let (rt, rr) = oneshot::channel();
    tx.send(ChatMessage::GetActive { reply_to: rt })
        .await
        .unwrap();
    assert!(
        rr.await.unwrap().is_some(),
        "default dialog should be active"
    );

    // A lookup with the wrong message type must return None.
    let wrong: Option<mpsc::Sender<MemoryGraphMessage>> = system.registry().get("chat");
    assert!(wrong.is_none());
}

// ============================================================================
// docs/examples.md → §3 Write a custom actor
// ============================================================================

pub enum PingMsg {
    Ping {
        reply_to: oneshot::Sender<Result<String, spire_core::actors::ActorError>>,
    },
}

pub struct PingActor {
    count: u32,
}

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

#[tokio::test]
async fn doc_example_3_custom_actor() {
    let system = ActorSystem::new();
    let (tx, _handle) = system.spawn(PingActor { count: 0 });

    let (reply_tx, reply_rx) = oneshot::channel();
    tx.send(PingMsg::Ping { reply_to: reply_tx }).await.unwrap();
    assert_eq!(reply_rx.await.unwrap().unwrap(), "pong #1");

    // The actor is stateful: a second ping increments the counter.
    let (reply_tx, reply_rx) = oneshot::channel();
    tx.send(PingMsg::Ping { reply_to: reply_tx }).await.unwrap();
    assert_eq!(reply_rx.await.unwrap().unwrap(), "pong #2");
}

// ============================================================================
// docs/examples.md → §4 ChildActor with a cached service sender
// ============================================================================

pub enum ProbeMsg {
    GetActive(oneshot::Sender<Option<ChatDialog>>),
}

pub struct ChatProbe {
    chat_tx: Option<mpsc::Sender<ChatMessage>>,
}

#[async_trait]
impl ChildActor for ChatProbe {
    type Message = ProbeMsg;

    fn init(&mut self, ctx: &mut ChildContext) {
        // Cache the sender once — never look up the registry per message.
        self.chat_tx = ctx.service("chat");
    }

    async fn handle(&mut self, _ctx: &ChildContext, msg: Self::Message) {
        let ProbeMsg::GetActive(reply) = msg;
        let (tx, rx) = oneshot::channel();
        let _ = self
            .chat_tx
            .as_ref()
            .unwrap()
            .send(ChatMessage::GetActive { reply_to: tx })
            .await;
        let _ = reply.send(rx.await.unwrap());
    }
}

#[tokio::test]
async fn doc_example_4_child_actor_with_cached_service() {
    let system = ActorSystem::new();
    let registry: Arc<ServiceRegistry> = system.registry().clone();

    // Register the chat actor, then spawn a ChildActor that resolves it in init.
    let (chat_tx, _handle) = system.spawn(ChatActor::new());
    registry.register("chat", chat_tx).unwrap();

    let probe_tx = spawn_child(&system, registry, |_ctx| ChatProbe { chat_tx: None });
    let (tx, rx) = oneshot::channel();
    probe_tx.send(ProbeMsg::GetActive(tx)).await.unwrap();
    let active = rx.await.unwrap();
    assert!(
        active.is_some(),
        "ChatProbe should resolve the default active dialog"
    );
}

// ============================================================================
// docs/examples.md → §5 Build a project file tree
// ============================================================================

#[tokio::test]
async fn doc_example_5_build_file_tree() {
    // A small temp project keeps the test hermetic (the docs example passes ".").
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"demo\"\n",
    )
    .unwrap();
    // Enough bytes that the size/50 line heuristic yields > 0 estimated lines.
    std::fs::write(
        dir.path().join("src/main.rs"),
        "fn main() {\n    // pad the file so the line estimate is non-zero\n    let answer = 42;\n    println!(\"{answer}\");\n}\n",
    )
    .unwrap();

    let tree = build_file_tree(dir.path(), /*no_ignore=*/ false);
    assert!(
        tree.total_file_count >= 2,
        "expected src/main.rs + Cargo.toml"
    );
    assert!(tree.directories.iter().any(|d| d.name == "src"));
    // The tree estimates lines from file sizes.
    assert!(tree.total_lines > 0);
}

// ============================================================================
// docs/examples.md → §6 Store and query a graph node
// ============================================================================

#[tokio::test]
async fn doc_example_6_store_and_query_graph_node() {
    let system = ActorSystem::new();
    let (mg_tx, _handle) = system.spawn(MemoryGraphActor::new());

    // 1. Initialize the store against a real data directory.
    let dir = tempfile::tempdir().unwrap();
    let (t, r) = oneshot::channel();
    mg_tx
        .send(MemoryGraphMessage::Initialize {
            data_dir: dir.path().to_path_buf(),
            reply_to: t,
        })
        .await
        .unwrap();
    r.await.unwrap().unwrap();

    // 2. Store an AttrNode (ids are opaque UUID strings — no format validation).
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
        .await
        .unwrap();
    let stored = r.await.unwrap().unwrap();
    assert_eq!(stored.name, "hello");
    assert_eq!(stored.node_type, "example");

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
        .await
        .unwrap();
    let nodes = r.await.unwrap().unwrap();
    assert!(
        nodes.iter().any(|n| n.id == "n1"),
        "expected the stored node in query results, got {nodes:?}"
    );
}

// ============================================================================
// docs/examples.md → §7 Spatial queries on the memory graph
// ============================================================================

#[tokio::test]
async fn doc_example_7_spatial_query() {
    let system = ActorSystem::new();
    let (mg_tx, _handle) = system.spawn(MemoryGraphActor::new());

    let dir = tempfile::tempdir().unwrap();
    let (t, r) = oneshot::channel();
    mg_tx
        .send(MemoryGraphMessage::Initialize {
            data_dir: dir.path().to_path_buf(),
            reply_to: t,
        })
        .await
        .unwrap();
    r.await.unwrap().unwrap();

    // A WGS84 point sensor, tagged with the AttrNode spatial helper.
    let now = chrono::Utc::now();
    let mut node = AttrNode {
        id: "s1".to_string(),
        node_type: "Sensor".to_string(),
        subtype: None,
        name: "sensor-nyc".to_string(),
        description: None,
        properties: Default::default(),
        embedding_id: None,
        created_at: now,
        updated_at: now,
        version: 1,
    };
    node.set_geo_point(geo::Point::new(-74.006, 40.7128)); // Point::new(lng, lat)
    let (t, r) = oneshot::channel();
    mg_tx
        .send(MemoryGraphMessage::StoreAttrNode { node, reply_to: t })
        .await
        .unwrap();
    r.await.unwrap().unwrap();

    // Intersects query over the slippy-map tile that contains the point.
    let (x, y) = spire_core::spatial::point_to_tile(&geo::Point::new(-74.006, 40.7128), 12);
    let (t, r) = oneshot::channel();
    mg_tx
        .send(MemoryGraphMessage::SpatialQuery {
            query: SpatialQuery::Intersects {
                geometry: geo::Geometry::Rect(spire_core::spatial::tile_bounds(12, x, y)),
            },
            node_type: None,
            subtype: None,
            limit: Some(50),
            reply_to: t,
        })
        .await
        .unwrap();
    let res = r.await.unwrap().unwrap();
    assert!(
        res.nodes.iter().any(|hit| hit.node.id == "s1"),
        "expected the sensor in its tile, got {} hits",
        res.total_results
    );
}

// ============================================================================
// docs/examples.md → §8 Query a RAG corpus
// ============================================================================

#[tokio::test]
async fn doc_example_8_query_rag_corpus() {
    let system = ActorSystem::new();
    let (mg_tx, _handle) = system.spawn(MemoryGraphActor::new());

    // The store must be initialized before queries can run.
    let dir = tempfile::tempdir().unwrap();
    let (t, r) = oneshot::channel();
    mg_tx
        .send(MemoryGraphMessage::Initialize {
            data_dir: dir.path().to_path_buf(),
            reply_to: t,
        })
        .await
        .unwrap();
    r.await.unwrap().unwrap();

    // Shared store: one graph for data plane + provenance.
    let embedder: Arc<dyn Embedder> = Arc::new(NoopEmbedder);
    let (rag_tx, _handle) = system.spawn(RagActor::new_shared(mg_tx, embedder));

    let (t, r) = oneshot::channel();
    rag_tx
        .send(RagMessage::Query {
            domain: "a7s".to_string(),
            query: "how does object detection work".to_string(),
            top_k: 5,
            reply_to: t,
        })
        .await
        .unwrap();

    let reply = r.await.unwrap();
    // Degraded-mode contract: the no-op embedder never produces vectors, so a
    // query with it must resolve to an error rather than silently returning
    // empty or fake results.
    assert!(
        reply.is_err(),
        "NoopEmbedder queries must fail loudly, got {reply:?}"
    );
    match &reply {
        Ok(chunks) => println!("doc_example_7: retrieval returned {} chunks", chunks.len()),
        Err(e) => println!("doc_example_7: retrieval failed without a real embedder: {e}"),
    }
}

// ============================================================================
// docs/examples.md → §9 Call a platform module
// ============================================================================

#[tokio::test]
async fn doc_example_9_call_platform_module() {
    let system = ActorSystem::new();
    let (fs_tx, _handle) = system.spawn(FilesystemModule::new());

    // Integration tests run with CWD = the crate root, so Cargo.toml exists.
    let (t, r) = oneshot::channel();
    fs_tx
        .send(FilesystemMessage::ReadFile {
            path: PathBuf::from("Cargo.toml"),
            reply_to: t,
        })
        .await
        .unwrap();
    match r.await.unwrap() {
        Ok(content) => assert!(
            content.contains("spire-core"),
            "Cargo.toml should name the crate, got: {content}"
        ),
        Err(e) => panic!("read failed: {e}"),
    }
}

// ============================================================================
// docs/examples.md → §10 Global config
// ============================================================================

#[tokio::test]
async fn doc_example_10_global_config() {
    // Serialize env-var mutation and point SPIRE_CONFIG_DIR at a temp dir so the
    // developer's real ~/.spire is never touched.
    let _guard = CONFIG_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let previous = std::env::var("SPIRE_CONFIG_DIR").ok();
    std::env::set_var("SPIRE_CONFIG_DIR", dir.path());

    // Persist a key (atomic temp-file + rename write).
    let result = spire_core::config::set_global_llm_config_key("deepseek.api_key", "sk-test");
    assert!(
        result.is_ok(),
        "set_global_llm_config_key failed: {result:?}"
    );

    // Read it back through both accessors.
    let key = spire_core::config::get_global_llm_config_key("deepseek.api_key");
    assert_eq!(key.as_deref(), Some("sk-test"));
    let cfg = spire_core::config::load_global_llm_config();
    assert_eq!(cfg.api_key, "sk-test");

    // Restore the previous env state.
    match previous {
        Some(v) => std::env::set_var("SPIRE_CONFIG_DIR", v),
        None => std::env::remove_var("SPIRE_CONFIG_DIR"),
    }
}

// ============================================================================
// docs/examples.md → §11 Vector tiles for the map UI
// ============================================================================

#[tokio::test]
async fn doc_example_11_vector_tiles() {
    let system = ActorSystem::new();
    let (mg_tx, _handle) = system.spawn(MemoryGraphActor::new());

    let dir = tempfile::tempdir().unwrap();
    let (t, r) = oneshot::channel();
    mg_tx
        .send(MemoryGraphMessage::Initialize {
            data_dir: dir.path().to_path_buf(),
            reply_to: t,
        })
        .await
        .unwrap();
    r.await.unwrap().unwrap();

    // A Singapore sensor; tile (807, 508) at z10 contains (103.85, 1.35).
    let now = chrono::Utc::now();
    let mut node = AttrNode {
        id: "s1".to_string(),
        node_type: "Sensor".to_string(),
        subtype: None,
        name: "sensor-sg".to_string(),
        description: None,
        properties: Default::default(),
        embedding_id: None,
        created_at: now,
        updated_at: now,
        version: 1,
    };
    node.set_geo_point(geo::Point::new(103.85, 1.35));
    let (t, r) = oneshot::channel();
    mg_tx
        .send(MemoryGraphMessage::StoreAttrNode { node, reply_to: t })
        .await
        .unwrap();
    r.await.unwrap().unwrap();

    // TileActor over the same graph.
    let (tile_tx, _th) = system.spawn(TileActor::new(mg_tx));

    // MVT bytes for the z10 tile.
    let filters = TileFilters {
        node_type: Some("Sensor".to_string()),
        ..Default::default()
    };
    let (t, r) = oneshot::channel();
    tile_tx
        .send(TileMessage::GetTile {
            filters,
            z: 10,
            x: 807,
            y: 508,
            reply_to: t,
        })
        .await
        .unwrap();
    let mvt = r.await.unwrap().unwrap();
    assert!(!mvt.is_empty(), "expected encoded MVT bytes");
}

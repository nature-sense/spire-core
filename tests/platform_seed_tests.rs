// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! End-to-end test for the platform/capability seeder
//! (`MemoryGraphMessage::BootstrapPlatforms`).
//!
//! The seeder is a **writer by construction**: this crate does not depend on spire-code, so it
//! cannot walk the registry's tree and consumes instead the flattened `capability_blocks` payload
//! that side attaches. Being a writer, it never reads back what it wrote — which makes it exactly
//! the kind of code that runs green while being wrong. A mis-keyed property or a wrong node label
//! would not fail anything; it would just produce a graph that silently verifies boards against
//! nothing. One defect of that kind has already been found here by hand: the platform delete
//! matched `'platform'` while the nodes are created `"Platform"`, so it deleted nothing.
//!
//! So this test states the contract in two halves, and the second half is the one the first
//! cannot cover: after re-bootstrapping with the blocks **removed**, nothing is left behind. A
//! stale capability is worse than a missing one, because a reader that trusts it is confidently
//! wrong.
//!
//! The payload is synthetic and deliberately small — the real registry is verified on the
//! spire-code side (its parse/validate tests). What is under test here is the *handler*: node
//! creation, dedup by path, the three edge kinds, the self-`via` skip, and the deletes.

use serde_json::json;
use spire_core::actors::{Actor, MemoryGraphActor, MemoryGraphMessage};
use spire_core::models::memory_graph::{AttrNode, GraphEdge};
use tokio::sync::{mpsc, oneshot};

/// Spawn a `MemoryGraphActor` over a fresh in-memory graph.
///
/// The store is made fresh rather than merely unique: `InitializeInMemory` recovers an existing
/// snapshot, so a leftover directory would make a re-run assert against the previous run's graph.
async fn spawn_graph() -> mpsc::Sender<MemoryGraphMessage> {
    let dir = std::env::temp_dir().join(format!("spire-platform-seed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let (tx, rx) = mpsc::channel(64);
    let _join = MemoryGraphActor::new().spawn(rx);
    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::InitializeInMemory {
        data_dir: dir,
        reply_to: t,
    })
    .await
    .expect("send init");
    r.await.expect("init reply").expect("init ok");
    tx
}

async fn bootstrap(tx: &mpsc::Sender<MemoryGraphMessage>, platforms: Vec<serde_json::Value>) {
    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::BootstrapPlatforms {
        platforms,
        reply_to: t,
    })
    .await
    .expect("send bootstrap");
    r.await.expect("bootstrap reply").expect("bootstrap ok");
}

async fn nodes_of_type(tx: &mpsc::Sender<MemoryGraphMessage>, node_type: &str) -> Vec<AttrNode> {
    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::QueryAttrNodes {
        node_type: Some(node_type.to_string()),
        subtype: None,
        name: None,
        limit: None,
        reply_to: t,
    })
    .await
    .expect("send query");
    r.await.expect("query reply").expect("query ok")
}

/// Every edge touching `id`, in both directions — capability edges point *at* a capability and
/// *at* a chip, so an outgoing-only read would miss half of them.
async fn edges_of(tx: &mpsc::Sender<MemoryGraphMessage>, id: &str) -> Vec<GraphEdge> {
    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::GetRelationships {
        node_id: id.to_string(),
        reply_to: t,
    })
    .await
    .expect("send rels");
    r.await.expect("rels reply").expect("rels ok")
}

/// `from -> to` pairs, as ids. Nodes are keyed by the id they were written with (the registry
/// path for a capability, the registry id for a platform), so these compare directly.
fn pairs(edges: &[GraphEdge]) -> Vec<(String, String)> {
    edges
        .iter()
        .map(|e| (e.from_id.clone(), e.to_id.clone()))
        .collect()
}

fn has_pair(edges: &[GraphEdge], from: &str, to: &str) -> bool {
    pairs(edges).iter().any(|(f, t)| f == from && t == to)
}

/// The properties of the `from -> to` edge, if it exists — the capability's values, stored on the
/// edge rather than on the shared capability node.
fn props_of(
    edges: &[GraphEdge],
    from: &str,
    to: &str,
) -> Option<std::collections::HashMap<String, serde_json::Value>> {
    edges
        .iter()
        .find(|e| e.from_id == from && e.to_id == to)
        .map(|e| e.properties.clone())
}

fn sorted_ids(nodes: &[AttrNode]) -> Vec<String> {
    let mut ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
    ids.sort();
    ids
}

/// A chip entry: it *is* its capabilities, so it declares paths and no `realizes` edges — but it
/// does **provide** them, and the silicon's own values ride on those `provides` edges.
fn chip(id: &str, paths: &[&str]) -> serde_json::Value {
    let provides: Vec<serde_json::Value> = paths
        .iter()
        .map(|p| json!({ "capability": p, "properties": { "silicon": id, "cores": 4 } }))
        .collect();
    json!({
        "id": id,
        "name": id,
        "properties": {},
        "capability_blocks": {
            "capabilities": paths,
            "provides": provides,
            "realizes": [],
            "carries": [],
        }
    })
}

/// The board, the C5 companion, and the P4 it is built around — the shape the registry actually
/// produces, including the two cases that are easy to get wrong: a path declared by *both* the
/// chip and the board (dedup), and a `via` that names the board's own chip (skipped, because an
/// edge from a thing to itself says nothing).
fn registry() -> Vec<serde_json::Value> {
    vec![
        chip(
            "esp32p4",
            &[
                "media.camera",
                "media.display",
                "media.video.encode",
                "compute.ml",
                "radio",
            ],
        ),
        chip("esp32c5", &["radio.wifi", "radio.bluetooth"]),
        json!({
            "id": "waveshare-esp32-p4-nano",
            "name": "Waveshare ESP32-P4-Nano",
            "properties": {"family": "esp32"},
            "capability_blocks": {
                "capabilities": ["media.display", "radio.wifi"],
                "realizes": [
                    {
                        "capability": "media.display",
                        "properties": {"via": "esp32p4", "interface": "spi"},
                    },
                    {
                        "capability": "radio.wifi",
                        "properties": {"via": "esp32c5", "standard": "802.11ax"},
                    },
                    // The tautology case: a `via` naming the board itself. Skipped, not written.
                    {"capability": "compute.ml", "properties": {"via": "waveshare-esp32-p4-nano"}},
                ],
                // The shape `seeder_input` actually produces: `{chip, properties}` — the chip id as
                // the edge's target, everything written beside it under `properties`.
                "carries": [
                    {"chip": "esp32c5", "properties": {"role": "radio", "link": {"bus": "sdio"}}}
                ],
                // Wiring flattens to one entry per declared **function** — the shape `seeder_input`
                // produces — for a `Pin` node and a `pins` edge each.
                "pins": [{"function": "i2c", "properties": {"sda": "GPIO7"}}],
            }
        }),
    ]
}

#[tokio::test]
async fn bootstrap_seeds_capability_nodes_and_edges() {
    let tx = spawn_graph().await;
    bootstrap(&tx, registry()).await;

    // ── Nodes ────────────────────────────────────────────────────────
    let platforms = nodes_of_type(&tx, "Platform").await;
    assert_eq!(
        sorted_ids(&platforms),
        vec!["esp32c5", "esp32p4", "waveshare-esp32-p4-nano"],
        "every entry becomes a Platform node, chips included"
    );

    // The union of every path, **deduped**: `media.display` is declared by both the chip and the
    // board and `radio.wifi` by both a chip and the board, so a seeder keying nodes on anything
    // but the path would show 11 here instead of 7.
    let caps = nodes_of_type(&tx, "Capability").await;
    assert_eq!(
        sorted_ids(&caps),
        vec![
            "compute.ml",
            "media.camera",
            "media.display",
            "media.video.encode",
            "radio",
            "radio.bluetooth",
            "radio.wifi",
        ],
        "one node per distinct capability path, deduped across chips and boards"
    );

    // ── Edges ────────────────────────────────────────────────────────
    let board = "waveshare-esp32-p4-nano";
    let board_edges = edges_of(&tx, board).await;
    assert!(
        has_pair(&board_edges, board, "media.display"),
        "board -realizes-> capability: {:?}",
        pairs(&board_edges)
    );
    assert!(
        has_pair(&board_edges, board, "esp32c5"),
        "board -carries-> companion chip: {:?}",
        pairs(&board_edges)
    );

    // The chip's own read sees both edge kinds that point at it: `carries` from the board and
    // `via` from the capability the chip provides.
    let chip_edges = edges_of(&tx, "esp32c5").await;
    assert!(
        has_pair(&chip_edges, board, "esp32c5"),
        "carries, seen from the chip: {:?}",
        pairs(&chip_edges)
    );
    assert!(
        has_pair(&chip_edges, "radio.wifi", "esp32c5"),
        "capability -via-> companion chip: {:?}",
        pairs(&chip_edges)
    );

    // A capability the board's *chip* provides gets a real `via` edge: the chip is a different
    // node from the board, so this says something the board's own `realizes` edge does not.
    let display_edges = edges_of(&tx, "media.display").await;
    assert!(
        has_pair(&display_edges, "media.display", "esp32p4"),
        "capability -via-> the chip that provides it: {:?}",
        pairs(&display_edges)
    );

    // A `via` naming the board *itself* is an edge from a thing to itself and says nothing, so the
    // seeder skips it rather than writing a tautology. Asserted from `compute.ml`, whose only
    // declared source is that self-reference.
    let ml_edges = edges_of(&tx, "compute.ml").await;
    assert!(
        !has_pair(&ml_edges, "compute.ml", board),
        "self-`via` must not be written: {:?}",
        pairs(&ml_edges)
    );

    // The predicate is the edge's meaning and must survive the reader **named**: an edge that
    // reads back as `Unknown` exists but cannot be asked for, and a capability graph whose edges
    // are findable only by shape is not one you can query for "what provides this".
    let kinds: std::collections::BTreeSet<String> = board_edges
        .iter()
        .map(|e| format!("{:?}", e.edge_type))
        .collect();
    assert!(
        kinds.contains("Realizes") && kinds.contains("Carries"),
        "edge types must name their predicate: {:?}",
        kinds
    );
}

/// The capability's **values** ride on the edge, not the node: a board's realization values
/// (`interface`) on `realizes`, a companion's (`role`, `link`) on `carries`, and a chip's own
/// (`silicon`, `cores`) on `provides`. Read back through `GraphEdge.properties`, they are the
/// values a caller queries for. A capability node is identity only — the same `media.camera` is
/// `spi` here and `mipi-csi` on a chip — so anything stored on the node would collide.
#[tokio::test]
async fn capability_values_are_stored_on_the_edges() {
    let tx = spawn_graph().await;
    bootstrap(&tx, registry()).await;

    // Board -realizes-> capability, carrying the board's own values. `via` is *not* among them:
    // it names the chip, and is the separate `via` edge.
    let board_edges = edges_of(&tx, "waveshare-esp32-p4-nano").await;
    let realizes = props_of(&board_edges, "waveshare-esp32-p4-nano", "media.display")
        .expect("the board realizes media.display");
    assert_eq!(realizes["interface"], json!("spi"));
    assert!(
        !realizes.contains_key("via"),
        "`via` is the edge's endpoint, not a stored property: {realizes:?}"
    );

    // Board -carries-> companion, with everything written beside the chip (minus its id). A nested
    // value is stored as a JSON-encoded scalar and restored to its structured form on the way back
    // (`selene_value_to_json`), so `link` is an object here, not a string.
    let carries = props_of(&board_edges, "waveshare-esp32-p4-nano", "esp32c5")
        .expect("the board carries esp32c5");
    assert_eq!(carries["role"], json!("radio"));
    assert_eq!(
        carries["link"],
        json!({ "bus": "sdio" }),
        "a nested value round-trips to its structured form: {carries:?}"
    );

    // Chip -provides-> capability, carrying the silicon's own values, and reading back **named** so
    // it can be asked for by kind rather than found by shape.
    let chip_edges = edges_of(&tx, "esp32p4").await;
    let provides =
        props_of(&chip_edges, "esp32p4", "compute.ml").expect("esp32p4 provides compute.ml");
    assert_eq!(provides["silicon"], json!("esp32p4"));
    assert_eq!(
        provides["cores"],
        json!(4),
        "a scalar keeps its type: {provides:?}"
    );
    let kinds: std::collections::BTreeSet<String> = chip_edges
        .iter()
        .map(|e| format!("{:?}", e.edge_type))
        .collect();
    assert!(
        kinds.contains("Provides"),
        "provides must read back named: {kinds:?}"
    );
}

/// A board's wiring becomes **its own nodes and edges**: one `Pin` node per declared function — the
/// node is the function, and the id is board-scoped so `led` on two boards is two functions — and a
/// `pins` edge carrying the assignment. The read side finds it by kind, the way it finds a
/// capability, instead of parsing a blob off the board.
/// **Every** platform fact survives the graph read — not a whitelist's worth. The graph is the
/// registry, so a read that named only some keys dropped the rest: `chip` (which entry is a board),
/// `rust_target` (what a build keys on), `library_hints` (what the fill prompt reads). A whitelist
/// in `platform_node_to_json` did exactly that.
#[tokio::test]
async fn every_platform_fact_survives_the_graph_read() {
    let tx = spawn_graph().await;
    let mut board = registry().pop().expect("the board");
    board["properties"] = json!({
        "os": "esp-idf",
        "family": "esp32",
        "chip": "esp32p4",
        "rust_target": "riscv32imafc-esp-espidf",
        "library_hints": "the board's own notes",
        "device_mcp_url": "http://x/mcp",
    });
    bootstrap(&tx, vec![board]).await;

    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::GetPlatforms { reply_to: t })
        .await
        .expect("send get");
    let nodes = r.await.expect("get reply").expect("get ok");
    let props = nodes[0]["properties"]
        .as_object()
        .expect("the platform node's properties");

    for key in [
        "family",
        "chip",
        "rust_target",
        "library_hints",
        "device_mcp_url",
    ] {
        assert!(
            props.contains_key(key),
            "`{key}` must survive the read: {props:?}"
        );
    }
    assert!(
        props.get("uuid").is_none(),
        "the envelope's own keys are not facts: {props:?}"
    );
}

#[tokio::test]
async fn wiring_is_its_own_node_and_edge() {
    let tx = spawn_graph().await;
    bootstrap(&tx, registry()).await;

    let pins = nodes_of_type(&tx, "Pin").await;
    assert_eq!(
        sorted_ids(&pins),
        vec!["waveshare-esp32-p4-nano/pins/i2c"],
        "one node per function, board-scoped"
    );
    assert_eq!(pins[0].name, "i2c");

    let edges = edges_of(&tx, "waveshare-esp32-p4-nano").await;
    let props = props_of(
        &edges,
        "waveshare-esp32-p4-nano",
        "waveshare-esp32-p4-nano/pins/i2c",
    )
    .expect("the board pins its i2c function");
    assert_eq!(props["sda"], json!("GPIO7"));

    let kinds: std::collections::BTreeSet<String> =
        edges.iter().map(|e| format!("{:?}", e.edge_type)).collect();
    assert!(
        kinds.contains("Pins"),
        "the predicate reads back named: {kinds:?}"
    );
}

#[tokio::test]
async fn re_bootstrap_without_blocks_leaves_nothing_behind() {
    let tx = spawn_graph().await;
    bootstrap(&tx, registry()).await;

    assert!(
        !nodes_of_type(&tx, "Capability").await.is_empty(),
        "the first bootstrap must seed something, or this test proves nothing"
    );

    // The registry no longer declares capabilities — a board whose blocks were removed. The
    // platform node is still seeded, so that node surviving is the rest of the handler still
    // working, in the same run where the capabilities must not survive.
    let bare = json!({
        "id": "waveshare-esp32-p4-nano",
        "name": "Waveshare ESP32-P4-Nano",
        "properties": {},
    });
    bootstrap(&tx, vec![bare]).await;

    assert!(
        nodes_of_type(&tx, "Capability").await.is_empty(),
        "capability nodes must not survive a bootstrap that declares none"
    );
    assert!(
        nodes_of_type(&tx, "Pin").await.is_empty(),
        "pin nodes must not survive a bootstrap that declares no wiring"
    );
    assert_eq!(
        sorted_ids(&nodes_of_type(&tx, "Platform").await),
        vec!["waveshare-esp32-p4-nano"],
        "the platform was re-seeded, and the chips now absent were not"
    );

    // Every edge died with its endpoint: `realizes` and `carries` with the removed chips and
    // capabilities, a `via` edge with the capability it left. A survivor here would be a dangling
    // truth — the exact failure this half of the test exists for.
    let survivors = edges_of(&tx, "waveshare-esp32-p4-nano").await;
    assert!(
        survivors.is_empty(),
        "no edge may outlive the nodes it connects: {:?}",
        pairs(&survivors)
    );
}

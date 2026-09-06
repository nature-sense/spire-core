// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! End-to-end test for `TileActor`: tile -> feature lookup over a real
//! `MemoryGraphActor`, exercising the spatial query + LRU cache pipeline.

use std::collections::HashSet;

use chrono::Utc;
use spire_core::actors::{
    Actor, MemoryGraphActor, MemoryGraphMessage, TileActor, TileFilters, TileMessage,
};
use spire_core::models::memory_graph::AttrNode;
use tokio::sync::{mpsc, oneshot};

/// Spawn a `MemoryGraphActor` initialized in `dir`.
async fn spawn_graph(dir: &std::path::Path) -> mpsc::Sender<MemoryGraphMessage> {
    std::fs::create_dir_all(dir).unwrap();
    let (tx, rx) = mpsc::channel(64);
    let _join = MemoryGraphActor::new().spawn(rx);
    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::Initialize {
        data_dir: dir.to_path_buf(),
        reply_to: t,
    })
    .await
    .expect("send init");
    r.await.expect("init reply").expect("init ok");
    tx
}

async fn store_point(
    tx: &mpsc::Sender<MemoryGraphMessage>,
    id: &str,
    node_type: &str,
    lng: f64,
    lat: f64,
) {
    let now = Utc::now();
    let mut node = AttrNode {
        id: id.to_string(),
        node_type: node_type.to_string(),
        subtype: None,
        name: id.to_string(),
        description: None,
        properties: Default::default(),
        embedding_id: None,
        created_at: now,
        updated_at: now,
        version: 1,
    };
    node.set_geo_point(geo::Point::new(lng, lat));
    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::StoreAttrNode { node, reply_to: t })
        .await
        .expect("send store");
    r.await.expect("store reply").expect("store ok");
}

async fn tile_bytes(
    tile_tx: &mpsc::Sender<TileMessage>,
    filters: TileFilters,
    z: u8,
    x: u32,
    y: u32,
) -> Vec<u8> {
    let (t, r) = oneshot::channel();
    tile_tx
        .send(TileMessage::GetTile {
            filters,
            z,
            x,
            y,
            reply_to: t,
        })
        .await
        .expect("send get-tile");
    r.await.expect("tile reply").expect("mvt bytes ok")
}

async fn tile_features(
    tile_tx: &mpsc::Sender<TileMessage>,
    filters: TileFilters,
    z: u8,
    x: u32,
    y: u32,
) -> HashSet<String> {
    let (t, r) = oneshot::channel();
    tile_tx
        .send(TileMessage::GetTileFeatures {
            filters,
            z,
            x,
            y,
            reply_to: t,
        })
        .await
        .expect("send tile request");
    let nodes = r.await.expect("tile reply").expect("tile features ok");
    nodes.iter().map(|n| n.id().to_string()).collect()
}

#[tokio::test]
async fn tile_actor_returns_features_for_a_tile() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mg_tx = spawn_graph(tmp.path()).await;

    // Spawn the TileActor on top of the graph.
    let (tile_tx, tile_rx) = mpsc::channel(64);
    let _join = TileActor::new(mg_tx.clone()).spawn(tile_rx);

    // Singapore cluster (tile z10: 807, 508 — verified in crate::spatial tests).
    store_point(&mg_tx, "s1", "Sensor", 103.850, 1.350).await;
    store_point(&mg_tx, "s2", "Sensor", 103.851, 1.352).await;
    store_point(&mg_tx, "c1", "Camera", 103.852, 1.353).await;
    // A far-away sensor (London) — must never appear in the Singapore tile.
    store_point(&mg_tx, "s-far", "Sensor", 2.000, 48.000).await;

    // Sensor filter: only the two Singapore sensors.
    let sensors = tile_features(
        &tile_tx,
        TileFilters {
            node_type: Some("Sensor".to_string()),
            ..Default::default()
        },
        10,
        807,
        508,
    )
    .await;
    assert_eq!(
        sensors,
        HashSet::from(["s1".to_string(), "s2".to_string()]),
        "sensor filter excludes the camera and the London sensor"
    );

    // No filter: camera joins the Singapore sensors; London still excluded.
    let all = tile_features(&tile_tx, TileFilters::default(), 10, 807, 508).await;
    assert_eq!(
        all,
        HashSet::from(["s1".to_string(), "s2".to_string(), "c1".to_string()])
    );

    // Repeat request hits the cache and returns the same set.
    let cached = tile_features(&tile_tx, TileFilters::default(), 10, 807, 508).await;
    assert_eq!(all, cached);

    // MVT bytes: the Sensor-only tile encodes with a "Sensor" layer and the
    // s1 id as a tag (protobuf stores both as their raw bytes).
    let sensor_filter = TileFilters {
        node_type: Some("Sensor".to_string()),
        ..Default::default()
    };
    let bytes = tile_bytes(&tile_tx, sensor_filter, 10, 807, 508).await;
    assert!(!bytes.is_empty(), "MVT payload present");
    let contains = |needle: &str| bytes.windows(needle.len()).any(|w| w == needle.as_bytes());
    assert!(contains("Sensor"), "Sensor layer name in bytes");
    assert!(contains("s1"), "feature id tag in bytes");

    // A neighbouring-but-empty tile yields no features and no payload.
    let empty = tile_features(&tile_tx, TileFilters::default(), 10, 807, 509).await;
    assert!(empty.is_empty());
}

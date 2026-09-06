// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! End-to-end tests for the memory graph's spatial query capability
//! (`MemoryGraphMessage::SpatialQuery`).
//!
//! Verifies all five predicates against a real `MemoryGraphActor` backed by an
//! in-memory WAL graph: bounding box, radius, k-nearest, contains, and
//! intersects — over both point nodes (`set_geo_point`) and full-polygon
//! nodes (`set_spatial_geometry`).

use std::collections::HashSet;

use chrono::Utc;
use spire_core::actors::{Actor, MemoryGraphActor, MemoryGraphMessage};
use spire_core::models::memory_graph::{AttrNode, DistanceScoredNode, SpatialQuery};
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

async fn store_node(tx: &mpsc::Sender<MemoryGraphMessage>, node: AttrNode) -> AttrNode {
    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::StoreAttrNode { node, reply_to: t })
        .await
        .expect("send store");
    r.await.expect("store reply").expect("store ok")
}

async fn spatial_query(
    tx: &mpsc::Sender<MemoryGraphMessage>,
    query: SpatialQuery,
) -> Vec<DistanceScoredNode> {
    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::SpatialQuery {
        query,
        node_type: None,
        subtype: None,
        limit: None,
        reply_to: t,
    })
    .await
    .expect("send spatial query");
    r.await
        .expect("spatial reply")
        .expect("spatial query ok")
        .nodes
}

fn point_node(id: &str, node_type: &str, name: &str, lng: f64, lat: f64) -> AttrNode {
    let now = Utc::now();
    let mut node = AttrNode {
        id: id.to_string(),
        node_type: node_type.to_string(),
        subtype: None,
        name: name.to_string(),
        description: None,
        properties: Default::default(),
        embedding_id: None,
        created_at: now,
        updated_at: now,
        version: 1,
    };
    node.set_geo_point(geo::Point::new(lng, lat));
    node
}

fn polygon_node(id: &str, name: &str, coords: &[(f64, f64)]) -> AttrNode {
    let now = Utc::now();
    let mut node = AttrNode {
        id: id.to_string(),
        node_type: "Zone".to_string(),
        subtype: Some("zone".to_string()),
        name: name.to_string(),
        description: None,
        properties: Default::default(),
        embedding_id: None,
        created_at: now,
        updated_at: now,
        version: 1,
    };
    let polygon = geo::Polygon::new(geo::LineString::from(coords.to_vec()), Vec::new());
    node.set_spatial_geometry(&geo::Geometry::Polygon(polygon));
    node
}

fn ids(nodes: &[DistanceScoredNode]) -> HashSet<String> {
    nodes.iter().map(|n| n.node.id().to_string()).collect()
}

#[tokio::test]
async fn spatial_queries_work_end_to_end() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let tx = spawn_graph(tmp.path()).await;

    // Sensors spread across North America + one far away in Europe.
    store_node(
        &tx,
        point_node("sensor-north", "Sensor", "north", -75.0, 45.0),
    )
    .await;
    store_node(
        &tx,
        point_node("sensor-center", "Sensor", "center", -72.0, 40.0),
    )
    .await;
    store_node(
        &tx,
        point_node("sensor-south", "Sensor", "south", -75.0, 35.0),
    )
    .await;
    store_node(
        &tx,
        point_node("sensor-europe", "Sensor", "europe", 2.0, 48.0),
    )
    .await;

    // A polygon "zone" wrapping lng -74..-70, lat 36..44.
    let zone_coords: [(f64, f64); 5] = [
        (-74.0, 36.0),
        (-70.0, 36.0),
        (-70.0, 44.0),
        (-74.0, 44.0),
        (-74.0, 36.0),
    ];
    store_node(&tx, polygon_node("zone-a", "alpha-zone", &zone_coords)).await;

    // ── BoundingBox ───────────────────────────────────────────────
    let rect = geo::Rect::new(
        geo::Coord { x: -76.0, y: 34.0 },
        geo::Coord { x: -69.0, y: 46.0 },
    );
    let hits = spatial_query(&tx, SpatialQuery::BoundingBox { rect }).await;
    assert_eq!(hits.len(), 4, "bbox should find the 3 sensors + zone");
    let got = ids(&hits);
    assert!(got.contains("sensor-north"));
    assert!(got.contains("sensor-center"));
    assert!(got.contains("sensor-south"));
    assert!(got.contains("zone-a"));
    assert!(!got.contains("sensor-europe"));

    // ── Radius ────────────────────────────────────────────────────
    let center = geo::Point::new(-72.0, 40.0); // inside zone-a
    let hits = spatial_query(
        &tx,
        SpatialQuery::Radius {
            center,
            radius_meters: 1_500_000.0, // ~13.5° — reaches the Americas sensors
        },
    )
    .await;
    let got = ids(&hits);
    assert!(got.contains("sensor-center"));
    assert!(got.contains("zone-a")); // distance 0: center inside the zone
    assert!(
        !got.contains("sensor-europe"),
        "europe should be out of range"
    );

    // ── Nearest ───────────────────────────────────────────────────
    let south_probe = geo::Point::new(-75.0, 30.0);
    let hits = spatial_query(
        &tx,
        SpatialQuery::Nearest {
            center: south_probe,
            k: 1,
        },
    )
    .await;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].node.id(), "sensor-south");
    assert!(hits[0].distance_meters.unwrap() > 0.0);

    // ── Contains ──────────────────────────────────────────────────
    // A point inside the zone returns the zone node only.
    let inside = geo::Point::new(-73.0, 40.0);
    let hits = spatial_query(
        &tx,
        SpatialQuery::Contains {
            geometry: geo::Geometry::Point(inside),
        },
    )
    .await;
    assert_eq!(ids(&hits), HashSet::from(["zone-a".to_string()]));

    // A smaller polygon fully inside the zone is contained by it.
    let inner_poly = geo::Polygon::new(
        geo::LineString::from(vec![
            (-73.5, 38.0),
            (-72.5, 38.0),
            (-72.5, 42.0),
            (-73.5, 42.0),
            (-73.5, 38.0),
        ]),
        Vec::new(),
    );
    let hits = spatial_query(
        &tx,
        SpatialQuery::Contains {
            geometry: geo::Geometry::Polygon(inner_poly),
        },
    )
    .await;
    assert_eq!(ids(&hits), HashSet::from(["zone-a".to_string()]));

    // A point outside returns nothing.
    let hits = spatial_query(
        &tx,
        SpatialQuery::Contains {
            geometry: geo::Geometry::Point(geo::Point::new(-60.0, 30.0)),
        },
    )
    .await;
    assert!(hits.is_empty());

    // ── Intersects ────────────────────────────────────────────────
    // Polygon wholly inside the zone (with no sensor in it) intersects it.
    let overlap_poly = geo::Polygon::new(
        geo::LineString::from(vec![
            (-72.5, 36.5),
            (-71.5, 36.5),
            (-71.5, 39.0),
            (-72.5, 39.0),
            (-72.5, 36.5),
        ]),
        Vec::new(),
    );
    let hits = spatial_query(
        &tx,
        SpatialQuery::Intersects {
            geometry: geo::Geometry::Polygon(overlap_poly),
        },
    )
    .await;
    assert_eq!(ids(&hits), HashSet::from(["zone-a".to_string()]));

    // A point inside the zone intersects the zone.
    let hits = spatial_query(
        &tx,
        SpatialQuery::Intersects {
            geometry: geo::Geometry::Point(inside),
        },
    )
    .await;
    assert_eq!(ids(&hits), HashSet::from(["zone-a".to_string()]));

    // A disjoint polygon matches nothing.
    let far_poly = geo::Polygon::new(
        geo::LineString::from(vec![
            (10.0, 10.0),
            (12.0, 10.0),
            (12.0, 12.0),
            (10.0, 12.0),
            (10.0, 10.0),
        ]),
        Vec::new(),
    );
    let hits = spatial_query(
        &tx,
        SpatialQuery::Intersects {
            geometry: geo::Geometry::Polygon(far_poly),
        },
    )
    .await;
    assert!(hits.is_empty());
}

#[tokio::test]
async fn spatial_query_respects_type_filter_and_limit() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let tx = spawn_graph(tmp.path()).await;

    store_node(&tx, point_node("s1", "Sensor", "a", -75.0, 40.0)).await;
    store_node(&tx, point_node("s2", "Sensor", "b", -74.0, 40.0)).await;
    store_node(&tx, point_node("c1", "Camera", "c", -73.0, 40.0)).await;

    let rect = geo::Rect::new(
        geo::Coord { x: -76.0, y: 39.0 },
        geo::Coord { x: -72.0, y: 41.0 },
    );
    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::SpatialQuery {
        query: SpatialQuery::BoundingBox { rect },
        node_type: Some("Sensor".to_string()),
        subtype: None,
        limit: Some(1),
        reply_to: t,
    })
    .await
    .expect("send");
    let result = r.await.expect("reply").expect("query ok");
    assert_eq!(result.nodes.len(), 1, "limit caps results");
    assert_eq!(result.total_results, 2, "total reflects unfiltered count");
    assert!(result.truncated, "truncated flag set when capped");
    assert!(result.nodes[0].node.is("Sensor"), "type filter applied");
}

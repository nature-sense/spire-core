# Spatial Queries on the Memory Graph

<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (c) 2026 NatureSense -->

## Spatial in one paragraph

Graph nodes can carry **WGS84 (EPSG:4326) geography** — a `latitude` /
`longitude` point, or an arbitrary geometry (polygon, line, multi-polygon) —
and `MemoryGraphActor` can answer spatial questions over them: *what is inside
this bounding box?*, *what is within 2 km of this sensor?*, *which zone contains
this point?*, *which features overlap this polygon?*. Location is stored as
plain `AttrNode` scalar properties, so no schema change is needed. Queries are
pre-filtered by a GQL range scan over scalar bounding-box columns, then refined
with exact predicates from the `geo` crate. Distances are geodesic
(**haversine, meters**).

## Overview

The spatial capability lives at three layers, mirroring how the rest of the
graph works:

| Layer | Where | Role |
| --- | --- | --- |
| Geometry functions | `crate::spatial` (`src/spatial.rs`) | Pure, synchronous math: distance, bounding boxes, `contains` / `intersects`, point-in-geometry. |
| Storage helpers | `AttrNode` (in `models::memory_graph`) | `set_geo_point`, `set_spatial_geometry`, `geo_point`, `geo_bounds`, `spatial_geometry` — write/read location on any node. |
| Query API | `MemoryGraphMessage::SpatialQuery` | Actor message that runs a `SpatialQuery` and returns a `SpatialQueryResult`. |

All three reuse the same node types and persistence path as the rest of the
memory graph (and the RAG KnowledgeStore), because both are the same
`MemoryGraphActor` under the hood.

## Coordinate system & units

- **WGS84 (EPSG:4326)**, the convention used by GeoJSON (RFC 7946).
- `geo::Point::new(longitude, latitude)` — **`x` is longitude, `y` is
  latitude**. GeoJSON coordinates are `[longitude, latitude]`, so a GeoJSON
  `[lon, lat]` pair maps directly onto `geo::Point::new(lon, lat)`.
- Distances are **geodesic meters** (haversine on a spherical Earth).
- `longitude`/`latitude` are the **singular** property names; bounding boxes
  use `min_lng`/`min_lat`/`max_lng`/`max_lat`.

## Storing location on nodes

Any `AttrNode` can carry location. There are two equivalent shapes; both are
queryable by every `SpatialQuery` variant:

| Shape | Properties | Set with |
| --- | --- | --- |
| Point | `latitude`, `longitude` (+ degenerate bounding box) | `AttrNode::set_geo_point` |
| Geometry | `geometry` (GeoJSON-serialized) + derived bounding box | `AttrNode::set_spatial_geometry` |
| (shared, derived) | `min_lng`, `min_lat`, `max_lng`, `max_lat` | written automatically by both helpers |
| (optional) | `altitude` | direct property write |

The bounding-box columns are what make GQL range pre-filtering possible — keep
them populated by using the helpers rather than writing raw JSON properties.
Set location **before** storing the node via
`MemoryGraphMessage::StoreAttrNode` / `MergeAttrNode`:

```rust
// A point (e.g. a sensor deployment).
let mut sensor = AttrNode { /* id, node_type "Sensor", name, … */ };
sensor.set_geo_point(geo::Point::new(-74.006, 40.7128));

// An arbitrary polygon feature (e.g. a survey zone) — GeoJSON decode is NOT
// required; build any geo::Geometry and store it.
let ring: geo::LineString<f64> = vec![(-74.5, 40.5), (-73.5, 40.5), (-73.5, 41.0), (-74.5, 41.0), (-74.5, 40.5)].into();
let mut zone = AttrNode { /* id, node_type "Zone", name, … */ };
zone.set_spatial_geometry(&geo::Geometry::Polygon(geo::Polygon::new(ring, Vec::new())));
```

## Query API

The message carries a `SpatialQuery` predicate plus optional `node_type` /
`subtype` filters and a result `limit`:

```rust
pub enum SpatialQuery {
    BoundingBox { rect: geo::Rect<f64> },          // node bbox fully inside rect
    Nearest     { center: geo::Point<f64>, k: usize }, // k closest, by meters
    Radius      { center: geo::Point<f64>, radius_meters: f64 },
    Contains    { geometry: geo::Geometry<f64> },  // node geometry contains this
    Intersects  { geometry: geo::Geometry<f64> },  // node geometry overlaps this
}

pub struct SpatialQueryResult {
    pub nodes: Vec<DistanceScoredNode>, // node + distance_meters (Radius/Nearest only)
    pub total_results: usize,
    pub truncated: bool,
}
```

`Contains` uses **boundary-inclusive** semantics (a point on a zone edge is
inside the zone); `Intersects` matches overlap, touching, or containment in
either direction. `Radius` treats a node whose geometry contains the center as
distance `0` (always within range).

## Examples

All examples assume a `MemoryGraphActor` is spawned and initialized as in
[`docs/examples.md`](examples.md) example 6, giving an `mg_tx` sender and a
`StoreAttrNode` path.

### Store a point, then run the five predicates

```rust
use spire_core::actors::MemoryGraphMessage;
use spire_core::models::memory_graph::{AttrNode, SpatialQuery};
use tokio::sync::oneshot;

async fn store(tx: &tokio::sync::mpsc::Sender<MemoryGraphMessage>, mut node: AttrNode) -> AttrNode {
    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::StoreAttrNode { node, reply_to: t }).await.unwrap();
    r.await.unwrap().unwrap()
}

// NYC coordinates: geo::Point::new(lng, lat).
let sensor = store(&mg_tx, AttrNode { /* node_type: "Sensor", … */ }).await;

async fn query(tx: &tokio::sync::mpsc::Sender<MemoryGraphMessage>, query: SpatialQuery) -> usize {
    let (t, r) = oneshot::channel();
    tx.send(MemoryGraphMessage::SpatialQuery { query, node_type: None, subtype: None,
        limit: Some(50), reply_to: t }).await.unwrap();
    let hits = r.await.unwrap().unwrap();
    println!("{} matches (truncated={})", hits.total_results, hits.truncated);
    hits.nodes.len()
}

// 1. Bounding box: nodes whose stored bbox is fully inside the rect.
let n = query(&mg_tx, SpatialQuery::BoundingBox {
    rect: geo::Rect::new(geo::Coord { x: -74.5, y: 40.5 }, geo::Coord { x: -73.5, y: 41.0 }),
}).await;

// 2. Radius: everything within 100 km of the sensor (meters).
let n = query(&mg_tx, SpatialQuery::Radius {
    center: geo::Point::new(-74.006, 40.7128),
    radius_meters: 100_000.0,
}).await;

// 3. K-nearest: the 3 closest nodes, sorted by geodesic distance.
let (t, r) = oneshot::channel();
mg_tx.send(MemoryGraphMessage::SpatialQuery {
    query: SpatialQuery::Nearest { center: geo::Point::new(-74.006, 40.7128), k: 3 },
    node_type: None, subtype: None, limit: None, reply_to: t,
}).await.unwrap();
for hit in r.await.unwrap().unwrap().nodes {
    println!("{} at {:.0} m", hit.node.name, hit.distance_meters.unwrap_or(0.0));
}

// 4. Contains: features whose geometry contains a point of interest.
let n = query(&mg_tx, SpatialQuery::Contains {
    geometry: geo::Geometry::Point(geo::Point::new(-73.99, 40.75)),
}).await;

// 5. Intersects: features overlapping a search polygon.
let search = geo::Polygon::new(
    vec![(-74.0, 40.6), (-73.8, 40.6), (-73.8, 40.9), (-74.0, 40.9), (-74.0, 40.6)].into(),
    Vec::new(),
);
let n = query(&mg_tx, SpatialQuery::Intersects {
    geometry: geo::Geometry::Polygon(search),
}).await;
```

### Read location back

```rust
let pt: Option<geo::Point<f64>> = sensor.geo_point();          // stored point
let rect: Option<geo::Rect<f64>> = sensor.geo_bounds();        // derived bbox
let geom: Option<geo::Geometry<f64>> = sensor.spatial_geometry(); // full geometry
let alt: Option<f64> = sensor.altitude();                      // optional meters
```

## Spatial + RAG (hybrid retrieval)

Spatial features and RAG corpora share the same store, so spatial filtering can
*complement* semantic retrieval. Two compositions cover most cases:

- **Spatial pre-filter → semantic re-rank** — run a `SpatialQuery` (e.g. a
  `Radius` or `Contains`) over geotagged `rag_chunk`/`rag_entity`-style nodes to
  narrow to a place, then embed and cosine-rank the survivors with the shared
  embedder (or ask `RagActor::Query` for the domain).
- **Semantic candidates → spatial validation** — retrieve `top_k` chunks with
  `RagActor::Query`, then keep only those whose node lies in the region of
  interest via a `Contains`/`BoundingBox` check.

There is no built-in "hybrid scorer" yet — compose the two calls in your
application. Nodes can carry **both** an embedding (`embedding_id` + vector via
`StoreNodeWithEmbedding`) **and** spatial properties, so one node can be found
semantically *and* geographically.

## Vector tiles for map UIs

The framework-side tile pipeline turns spatial features into slippy-map tiles:

- `TileActor` (`src/actors/tile.rs`) — answers `GetTileFeatures` (the `AttrNode`s
  intersecting tile `z/x/y`, LRU-cached per filters) and `GetTile` (MVT bytes).
- `crate::tiles::encode_tile` — pure MVT encoding: WGS84 -> tile-local
  (Web Mercator, extent 4096) -> clipped protobuf bytes, one layer per `node_type`.
- `crate::spatial::{tile_bounds, point_to_tile, lonlat_to_tile_coord}` — the
  projection math between tiles and lon/lat.

```rust
// MVT bytes for the z10 tile (x=807, y=508) around Singapore (103.85, 1.35).
let (t, r) = oneshot::channel();
tile_tx
    .send(TileMessage::GetTile {
        filters: TileFilters { node_type: Some("Sensor".to_string()), ..Default::default() },
        z: 10, x: 807, y: 508, reply_to: t,
    })
    .await.unwrap();
let mvt_bytes: Vec<u8> = r.await.unwrap().unwrap();
```

Features are cached per `(filters, z, x, y)`, so panning a viewport queries the
store once per tile; the CPU-bound MVT encode runs on a blocking task off the
actor mailbox. Geometry crossing a tile edge is clipped to the extent, so a
feature spanning many tiles is drawn correctly in each. See
[`docs/examples.md`](examples.md) for the full actor wiring.

## Best practices

- **Always use the `AttrNode` helpers** (`set_geo_point` / `set_spatial_geometry`)
  so the scalar `min_*`/`max_*` bounding-box columns stay in sync — those
  columns are what the GQL pre-filter scans.
- **Keep coordinates as scalar numbers**, not JSON strings — range predicates
  (`WHERE n.latitude >= …`) only work on native numeric properties.
- **Remember the axis order**: `Point::new(lng, lat)`, `Coord { x: lng, y: lat }`.
- **Use meters** for radii and distances (`Radius.radius_meters`,
  `Nearest` results report `distance_meters`).
- **`Contains` is boundary-inclusive**; a point exactly on a ring counts.
- **Filter early, refine late** — pass `node_type`/`subtype` in the message and
  keep `limit` small; the actor already pre-filters by bounding box before
  running exact predicates.
- **Querying does not need GeoJSON** — store a `geo::Geometry` directly; parsing
  GeoJSON documents is an ingestion concern, not a query concern.

## Where to go next

- [`docs/examples.md`](examples.md) — runnable graph + spatial examples
- [`docs/api.md`](api.md) — `spatial` module and `MemoryGraphMessage::SpatialQuery` signatures
- [`docs/rag.md`](rag.md) — the RAG knowledge store these nodes can live in
- [`src/spatial.rs`](../src/spatial.rs) — the operation functions (fully unit-tested)
- [`tests/spatial_query_tests.rs`](../tests/spatial_query_tests.rs) — end-to-end coverage

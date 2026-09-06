# Spatial Queries & Vector Tiles

<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (c) 2026 NatureSense -->

## Spatial in one paragraph

Graph nodes can carry **WGS84 (EPSG:4326) geography** — a `latitude`/
`longitude` point or an arbitrary geometry (polygon, line, multi-polygon) — and
`MemoryGraphActor` answers spatial questions over them: *what is inside this
bounding box?*, *what is within 2 km of this sensor?*, *which zone contains this
point?*, *which features overlap this polygon?*. Location is stored as plain
`AttrNode` scalar properties, so no schema change is needed. Queries pre-filter
with a GQL range scan over scalar bounding-box columns, then refine with exact
predicates from the `geo` crate. Distances are geodesic (**haversine, meters**).

## Vector tiles in one paragraph

The same store feeds map UIs. A slippy-map tile `z/x/y` (Web Mercator,
EPSG:3857) maps to a WGS84 window via `crate::spatial::tile_bounds`; `TileActor`
answers `GetTileFeatures` (the `AttrNode`s intersecting that window, LRU-cached
per `(filters, z, x, y)`) and `GetTile` (the features encoded to **MVT bytes** by
`crate::tiles::encode_tile`). Geometry is clipped to the tile extent, so a
feature spanning many tiles is drawn correctly in each.

## Coordinate system & units

- **WGS84 (EPSG:4326)**, the convention used by GeoJSON (RFC 7946).
- `geo::Point::new(longitude, latitude)` — **`x` is longitude, `y` is latitude**.
- Distances are **geodesic meters** (haversine).
- Tiles use **Web Mercator (EPSG:3857)**; the conversion is pure math in
  `crate::spatial` (no projection library needed).

## Storing location on nodes

Any `AttrNode` can carry location. Two shapes are stored; both are queryable by
every `SpatialQuery` variant:

| Shape | Property keys | Set with |
| --- | --- | --- |
| Point | `latitude`, `longitude` (+ degenerate bounding box) | `AttrNode::set_geo_point(Point)` |
| Geometry | `geometry` (GeoJSON-serialized) + derived bounding box | `AttrNode::set_spatial_geometry(&Geometry)` |
| Derived bbox | `min_lng`, `min_lat`, `max_lng`, `max_lat` | written automatically by both helpers |
| Optional | `altitude` | direct property write |

The bounding-box columns drive the GQL range pre-filter — always use the helpers
rather than hand-writing JSON properties. Set location **before** storing via
`MemoryGraphMessage::StoreAttrNode` / `MergeAttrNode`.

```rust
// A point (e.g. a sensor deployment).
let mut sensor = AttrNode { /* id, node_type "Sensor", name, ... */ };
sensor.set_geo_point(geo::Point::new(-74.006, 40.7128));

// An arbitrary polygon feature (e.g. a survey zone).
let ring: geo::LineString<f64> = vec![
    (-74.5, 40.5), (-73.5, 40.5), (-73.5, 41.0), (-74.5, 41.0), (-74.5, 40.5),
].into();
let mut zone = AttrNode { /* id, node_type "Zone", name, ... */ };
zone.set_spatial_geometry(&geo::Geometry::Polygon(geo::Polygon::new(ring, Vec::new())));
```

Read location back with `geo_point()`, `geo_bounds()`, `spatial_geometry()` and
`altitude()`.

## Spatial query API

`MemoryGraphMessage::SpatialQuery` carries a predicate plus optional
`node_type`/`subtype` filters and a result `limit`:

```rust
pub enum SpatialQuery {
    BoundingBox { rect: geo::Rect<f64> },          // node bbox fully inside rect
    Nearest     { center: geo::Point<f64>, k: usize }, // k closest, by meters
    Radius      { center: geo::Point<f64>, radius_meters: f64 },
    Contains    { geometry: geo::Geometry<f64> },  // node geometry contains this
    Intersects  { geometry: geo::Geometry<f64> },  // node geometry overlaps this
}

pub struct SpatialQueryResult {
    pub nodes: Vec<DistanceScoredNode>, // node + distance_meters (Radius/Nearest)
    pub total_results: usize,
    pub truncated: bool,
}
```

Semantics:

- `BoundingBox` — nodes whose stored bounding box lies fully inside `rect`.
- `Radius` / `Nearest` — geodesic distance from `center` to the node's geometry;
  a node whose geometry contains the center is at distance `0`. Results are
  sorted ascending and report `distance_meters`.
- `Contains` — **boundary-inclusive** point-in-zone; a point on the edge counts.
- `Intersects` — overlap, touching, or containment in either direction.

## Tile projection helpers

`crate::spatial` exposes the slippy-map math used by the tile pipeline:

```rust
pub const MAX_MERCATOR_LATITUDE: f64;
pub fn tile_bounds(z: u8, x: u32, y: u32) -> geo::Rect<f64>;   // tile -> lon/lat window
pub fn point_to_tile(p: &geo::Point<f64>, z: u8) -> (u32, u32); // feature -> tile
pub fn lonlat_to_tile_coord(lon, lat, z, x, y, extent) -> (f64, f64); // -> tile-local px
```

`lonlat_to_tile_coord` with extent `4096` is exactly what `encode_tile` feeds the
MVT encoder, so tiles line up with the same queries the spatial API runs.

## Vector tiles for map UIs

- **`TileActor`** (`src/actors/tile.rs`) — `GetTileFeatures` returns the
  `AttrNode`s intersecting tile `z/x/y` (from the spatial query, cached);
  `GetTile` returns **MVT bytes**. `ClearCache` empties the LRU.
- **`crate::tiles::encode_tile`** — pure encoding: group by `node_type` into
  layers, project to tile-local space, clip to the extent, encode with the
  `mvt` crate. Each feature carries `id` + `name` tags plus scalar `properties`.
- **Cache** — an LRU keyed by `(filters, z, x, y)` inside `TileActor`, so
  panning a viewport queries the store once per tile. Encoding runs on a
  blocking task off the mailbox.

```rust
use spire_core::actors::{TileActor, TileFilters, TileMessage};
use tokio::sync::oneshot;

let (tile_tx, _h) = /* system.spawn(TileActor::new(mg_tx)) */;

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

## Spatial + RAG (hybrid retrieval)

Spatial features and RAG corpora share the same store, so spatial filtering can
complement semantic retrieval:

- **Spatial pre-filter → semantic re-rank** — run a `SpatialQuery` (e.g. a
  `Radius` or `Contains`) over geotagged `rag_chunk`/`rag_entity`-style nodes to
  narrow to a place, then embed and cosine-rank the survivors.
- **Semantic candidates → spatial validation** — retrieve `top_k` chunks with
  `RagActor::Query`, then keep only those whose node lies in the region of
  interest.

There is no built-in hybrid scorer yet — compose the two calls in your
application. A node can carry **both** an embedding (`StoreNodeWithEmbedding`)
**and** spatial properties, so it can be found semantically *and* geographically.

## Best practices

- **Always use the `AttrNode` helpers** so the scalar `min_*`/`max_*`
  bounding-box columns stay in sync — those columns are what the GQL pre-filter
  scans.
- **Keep coordinates as scalar numbers**, not JSON strings — range predicates
  (`WHERE n.latitude >= ...`) only work on native numeric properties.
- **Remember the axis order**: `Point::new(lng, lat)`, `Coord { x: lng, y: lat }`.
- **Use meters** for radii/distances (`radius_meters`, `distance_meters`).
- **`Contains` is boundary-inclusive**; a point exactly on a ring counts.
- **Filter early, refine late** — pass `node_type`/`subtype` and keep `limit`
  small; the actor pre-filters by bounding box before running exact predicates.
- **Store geometry, not GeoJSON** — `set_spatial_geometry` takes a
  `geo::Geometry`; decoding GeoJSON documents is an ingestion concern.
- **Cache tiles**, never re-query the same `(filters, z, x, y)`; `TileActor`
  already does this.

## Examples & where to go next

- [`docs/examples.md`](examples.md) — runnable graph, spatial, and tile examples
- [`docs/api.md`](api.md) — `spatial` / `tiles` modules and message signatures
- [`src/spatial.rs`](../src/spatial.rs) — geometry functions + slippy-map math
- [`src/tiles.rs`](../src/tiles.rs) — MVT encoder
- [`src/actors/tile.rs`](../src/actors/tile.rs) — tile actor + LRU cache
- [`tests/spatial_query_tests.rs`](../tests/spatial_query_tests.rs) and
  [`tests/tile_actor_tests.rs`](../tests/tile_actor_tests.rs) — end-to-end coverage

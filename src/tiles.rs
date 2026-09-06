// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! MVT (Mapbox Vector Tile) encoding from memory-graph features.
//!
//! Pure, stateless functions that translate `AttrNode` features into encoded
//! vector-tile bytes for a slippy-map tile `z`/`x`/`y`. Composition is
//! intentional:
//!
//! ```text
//! SpatialQuery (graph actor) -> AttrNode features
//!   -> crate::tiles::encode_tile  (WGS84 -> tile-local, clip, encode)
//!   -> MVT bytes for the UI
//! ```
//!
//! Coordinates are projected from WGS84 (EPSG:4326) to the tile-local Web
//! Mercator space (EPSG:3857, extent 4096) via `crate::spatial`, then encoded
//! with the `mvt` crate. Geometry that crosses a tile edge is clipped to the
//! tile extent by the encoder.

use anyhow::{anyhow, Result};
use geo::{Geometry, Polygon};
use mvt::{GeomData, GeomEncoder, GeomType, Tile};

use crate::models::memory_graph::AttrNode;
use crate::spatial::lonlat_to_tile_coord;

/// Standard MVT extent: the tile is a `TILE_EXTENT x TILE_EXTENT` grid.
pub const TILE_EXTENT: u32 = 4096;

/// Fallback layer name when a node's `node_type` is empty.
const DEFAULT_LAYER: &str = "features";

/// The tile-local bounding box used to clip geometry to the tile.
fn tile_bbox() -> pointy::BBox<f64> {
    let e = TILE_EXTENT as f64;
    pointy::BBox::new([(0.0_f64, 0.0_f64), (e, e)])
}

/// Encode `features` into a vector tile for `z`/`x`/`y`.
///
/// Features are grouped into one MVT layer per `node_type`. Each feature
/// carries `id` and `name` tags plus any scalar entries from its `properties`
/// map (nested values are skipped). Returns the encoded tile bytes.
pub fn encode_tile(features: &[AttrNode], z: u8, x: u32, y: u32) -> Result<Vec<u8>> {
    let mut tile = Tile::new(TILE_EXTENT);

    // Group by node type, preserving a stable (sorted) layer order.
    let mut by_type: std::collections::BTreeMap<&str, Vec<&AttrNode>> =
        std::collections::BTreeMap::new();
    for node in features {
        let name = if node.node_type.is_empty() {
            DEFAULT_LAYER
        } else {
            node.node_type.as_str()
        };
        by_type.entry(name).or_default().push(node);
    }

    for (layer_name, nodes) in by_type {
        let mut layer = tile.create_layer(layer_name);
        for node in nodes {
            let Some(geom) = feature_geometry(node, z, x, y)? else {
                continue; // unsupported / empty geometry
            };
            let mut feature = layer.into_feature(geom);
            add_tags(&mut feature, node);
            layer = feature.into_layer();
        }
        tile.add_layer(layer)?;
    }

    tile.to_bytes()
        .map_err(|e| anyhow!("mvt encode failed: {e}"))
}

/// The `GeomData` for a node: its full stored geometry when present, else its
/// `latitude`/`longitude` point. `None` for unsupported or empty geometry.
fn feature_geometry(node: &AttrNode, z: u8, x: u32, y: u32) -> Result<Option<GeomData>> {
    let geometry: Geometry<f64> = match node.spatial_geometry() {
        Some(g) => g,
        None => match node.geo_point() {
            Some(p) => Geometry::Point(p),
            None => return Ok(None),
        },
    };
    encode_geometry(&geometry, z, x, y)
}

/// Project one WGS84 lon/lat coordinate into tile-local space.
fn proj(lon: f64, lat: f64, z: u8, x: u32, y: u32) -> (f64, f64) {
    lonlat_to_tile_coord(lon, lat, z, x, y, TILE_EXTENT as f64)
}

/// Encode a `geo::Geometry` as MVT geometry data for the tile.
///
/// Multi geometries become a single MVT feature of the corresponding base
/// type (`MultiPoint -> Point`, `MultiLineString -> Linestring`,
/// `MultiPolygon -> Polygon`). `GeometryCollection` and `Triangle` are not
/// encoded (return `None`) — rare in map data.
fn encode_geometry(g: &Geometry<f64>, z: u8, x: u32, y: u32) -> Result<Option<GeomData>> {
    let bbox = tile_bbox();
    match g {
        Geometry::Point(p) => encode_points(&[proj(p.x(), p.y(), z, x, y)], bbox),
        Geometry::MultiPoint(mp) => {
            let pts: Vec<(f64, f64)> = mp.0.iter().map(|p| proj(p.x(), p.y(), z, x, y)).collect();
            encode_points(&pts, bbox)
        }
        Geometry::Line(line) => {
            let pts = vec![
                proj(line.start.x, line.start.y, z, x, y),
                proj(line.end.x, line.end.y, z, x, y),
            ];
            encode_linestring(&pts, bbox)
        }
        Geometry::LineString(ls) => {
            let pts: Vec<(f64, f64)> = ls.0.iter().map(|c| proj(c.x, c.y, z, x, y)).collect();
            encode_linestring(&pts, bbox)
        }
        Geometry::MultiLineString(mls) => {
            let lines: Vec<Vec<(f64, f64)>> = mls
                .0
                .iter()
                .map(|ls| ls.0.iter().map(|c| proj(c.x, c.y, z, x, y)).collect())
                .collect();
            encode_multilinestring(&lines, bbox)
        }
        Geometry::Polygon(poly) => {
            let mut rings = polygon_rings(poly, z, x, y);
            encode_polygon_rings(&mut rings, bbox)
        }
        Geometry::MultiPolygon(mp) => {
            let mut rings: Vec<(Vec<(f64, f64)>, bool)> = Vec::new();
            for poly in &mp.0 {
                let mut own = polygon_rings(poly, z, x, y);
                rings.append(&mut own);
            }
            encode_polygon_rings(&mut rings, bbox)
        }
        Geometry::Rect(r) => {
            // Four lon/lat corners; winding is normalized downstream.
            let corners = [
                (r.min().x, r.min().y),
                (r.max().x, r.min().y),
                (r.max().x, r.max().y),
                (r.min().x, r.max().y),
            ];
            let ring: Vec<(f64, f64)> = corners
                .iter()
                .map(|(lon, lat)| proj(*lon, *lat, z, x, y))
                .collect();
            let mut rings = vec![(ring, true)];
            encode_polygon_rings(&mut rings, bbox)
        }
        // Rarely relevant to tiles; skip rather than guess.
        Geometry::Triangle(_) | Geometry::GeometryCollection(_) => Ok(None),
    }
}

/// A polygon's rings as `(points, is_exterior)` pairs, projected to tile-local
/// coordinates: the exterior ring first, then any holes.
fn polygon_rings(poly: &Polygon<f64>, z: u8, x: u32, y: u32) -> Vec<(Vec<(f64, f64)>, bool)> {
    let mut rings = Vec::new();
    let exterior: Vec<(f64, f64)> = poly
        .exterior()
        .0
        .iter()
        .map(|c| proj(c.x, c.y, z, x, y))
        .collect();
    rings.push((exterior, true));
    for hole in poly.interiors() {
        let pts: Vec<(f64, f64)> = hole.0.iter().map(|c| proj(c.x, c.y, z, x, y)).collect();
        rings.push((pts, false));
    }
    rings
}

/// Signed shoelace area of a ring in tile-local coordinates. Exterior rings
/// are normalized to positive area, holes to negative (the MVT convention).
fn signed_area(ring: &[(f64, f64)]) -> f64 {
    let mut sum = 0.0;
    for i in 0..ring.len() {
        let (x1, y1) = ring[i];
        let (x2, y2) = ring[(i + 1) % ring.len()];
        sum += x1 * y2 - x2 * y1;
    }
    sum / 2.0
}

/// Reverse a ring's points when its winding does not match the expectation.
fn ensure_winding(ring: &mut [(f64, f64)], want_exterior: bool) {
    if ring.len() < 3 {
        return;
    }
    let positive = signed_area(ring) > 0.0;
    if positive != want_exterior {
        ring.reverse();
    }
}

fn encode_points(pts: &[(f64, f64)], bbox: pointy::BBox<f64>) -> Result<Option<GeomData>> {
    if pts.is_empty() {
        return Ok(None);
    }
    let mut enc = GeomEncoder::new(GeomType::Point).bbox(bbox);
    for &(x, y) in pts {
        enc = enc.point(x, y)?;
    }
    Ok(Some(enc.encode()?))
}

fn encode_linestring(pts: &[(f64, f64)], bbox: pointy::BBox<f64>) -> Result<Option<GeomData>> {
    if pts.len() < 2 {
        return Ok(None);
    }
    let mut enc = GeomEncoder::new(GeomType::Linestring).bbox(bbox);
    for &(x, y) in pts {
        enc = enc.point(x, y)?;
    }
    Ok(Some(enc.encode()?))
}

fn encode_multilinestring(
    lines: &[Vec<(f64, f64)>],
    bbox: pointy::BBox<f64>,
) -> Result<Option<GeomData>> {
    let mut enc = GeomEncoder::new(GeomType::Linestring).bbox(bbox);
    let mut any = false;
    for line in lines {
        if line.len() < 2 {
            continue;
        }
        for &(x, y) in line {
            enc = enc.point(x, y)?;
        }
        enc = enc.complete()?;
        any = true;
    }
    if !any {
        return Ok(None);
    }
    Ok(Some(enc.encode()?))
}

/// Encode polygon rings as a single MVT polygon feature. The first ring must
/// be the exterior; subsequent rings alternate by winding (holes negative).
fn encode_polygon_rings(
    rings: &mut [(Vec<(f64, f64)>, bool)],
    bbox: pointy::BBox<f64>,
) -> Result<Option<GeomData>> {
    if rings.is_empty() {
        return Ok(None);
    }
    let mut enc = GeomEncoder::new(GeomType::Polygon).bbox(bbox);
    let mut any = false;
    for (ring, want_exterior) in rings {
        if ring.len() < 3 {
            continue;
        }
        ensure_winding(ring, *want_exterior);
        for &(x, y) in ring.iter() {
            enc = enc.point(x, y)?;
        }
        enc = enc.complete()?;
        any = true;
    }
    if !any {
        return Ok(None);
    }
    Ok(Some(enc.encode()?))
}

/// Add `id` + `name` tags and the node's scalar `properties` to a feature.
fn add_tags(feature: &mut mvt::Feature, node: &AttrNode) {
    feature.add_tag_string("id", node.id());
    feature.add_tag_string("name", node.name());
    for (key, value) in &node.properties {
        if key == "id" || key == "name" {
            continue; // already tagged above
        }
        match value {
            serde_json::Value::String(s) => feature.add_tag_string(key, s),
            serde_json::Value::Bool(b) => feature.add_tag_bool(key, *b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    feature.add_tag_int(key, i);
                } else if let Some(u) = n.as_u64() {
                    feature.add_tag_uint(key, u);
                } else if let Some(f) = n.as_f64() {
                    feature.add_tag_double(key, f);
                }
            }
            _ => {} // skip nested values / null
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use geo::{Geometry, LineString, Point, Polygon};

    fn node(id: &str, name: &str, node_type: &str) -> AttrNode {
        AttrNode {
            id: id.to_string(),
            node_type: node_type.to_string(),
            subtype: None,
            name: name.to_string(),
            description: None,
            properties: Default::default(),
            embedding_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            version: 1,
        }
    }

    fn point_node(id: &str, lng: f64, lat: f64) -> AttrNode {
        let mut n = node(id, id, "Sensor");
        n.set_geo_point(Point::new(lng, lat));
        n
    }

    fn bytes_contain(haystack: &[u8], needle: &str) -> bool {
        haystack
            .windows(needle.len())
            .any(|w| w == needle.as_bytes())
    }

    #[test]
    fn encode_tile_point_contains_layer_and_id() {
        // Singapore (103.85, 1.35) sits in tile (807, 508) at z10.
        let features = vec![point_node("sensor-1", 103.85, 1.35)];
        let bytes = encode_tile(&features, 10, 807, 508).expect("encode");
        assert!(!bytes.is_empty());
        assert!(bytes_contain(&bytes, "Sensor"), "layer name encoded");
        assert!(bytes_contain(&bytes, "sensor-1"), "feature id tag encoded");
    }

    #[test]
    fn encode_tile_empty_features_is_empty_tile() {
        let bytes = encode_tile(&[], 10, 807, 508).expect("encode");
        assert!(bytes.is_empty(), "no layers -> empty payload");
    }

    #[test]
    fn encode_tile_polygon() {
        let mut zone = node("zone-1", "alpha", "Zone");
        let ring: LineString<f64> = vec![
            (103.80, 1.30),
            (103.90, 1.30),
            (103.90, 1.40),
            (103.80, 1.40),
            (103.80, 1.30),
        ]
        .into();
        let hole: LineString<f64> = vec![
            (103.84, 1.34),
            (103.86, 1.34),
            (103.86, 1.36),
            (103.84, 1.36),
            (103.84, 1.34),
        ]
        .into();
        zone.set_spatial_geometry(&Geometry::Polygon(Polygon::new(ring, vec![hole])));

        let bytes = encode_tile(&[zone], 10, 807, 508).expect("encode");
        assert!(!bytes.is_empty());
        assert!(bytes_contain(&bytes, "Zone"));
        assert!(bytes_contain(&bytes, "zone-1"));
    }
}

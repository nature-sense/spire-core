// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Spatial geometry operations for memory-graph queries.
//!
//! This module is the "functions" layer behind
//! `MemoryGraphMessage::SpatialQuery`: geodesic distance, bounding boxes,
//! radius/nearest geometry and the `contains` / `intersects` predicates. All
//! functions are pure and synchronous, so they are trivial to unit-test and
//! reusable outside the graph actor.
//!
//! # Coordinate system
//!
//! Coordinates follow the WGS84 (EPSG:4326) longitude/latitude convention
//! (matching GeoJSON RFC 7946): a `geo::Point::new(longitude, latitude)` has
//! `x = longitude` and `y = latitude`. Geodesic distances are returned in
//! **meters** (haversine on a spherical Earth).
//!
//! # Node storage convention
//!
//! Graph nodes express their location via the following property keys (see
//! the `AttrNode` helpers `geo_point` / `spatial_geometry` in
//! `models::memory_graph`):
//!
//! | Key | Value |
//! | --- | --- |
//! | `latitude` / `longitude` | point coordinates (decimal degrees) |
//! | `altitude` | optional elevation, meters |
//! | `min_lat` / `min_lng` / `max_lat` / `max_lng` | pre-computed axis-aligned bounding box |
//! | `geometry` | optional full geometry, serialized as GeoJSON |
//!
//! The bounding-box columns let the actor pre-filter with a GQL `WHERE`
//! range scan before refining with the exact predicates here.

use geo::{
    BoundingRect, Closest, ClosestPoint, Contains, Coord, CoordsIter, Distance, Geometry,
    Haversine, Intersects, LinesIter, Point, Rect,
};

pub use geo;

// ============================================================================
// Node property keys
// ============================================================================

/// Property key for the node latitude in decimal degrees.
pub const PROP_LATITUDE: &str = "latitude";
/// Property key for the node longitude in decimal degrees.
pub const PROP_LONGITUDE: &str = "longitude";
/// Property key for the node altitude in meters (optional).
pub const PROP_ALTITUDE: &str = "altitude";
/// Property key for the bounding-box minimum longitude.
pub const PROP_MIN_LNG: &str = "min_lng";
/// Property key for the bounding-box minimum latitude.
pub const PROP_MIN_LAT: &str = "min_lat";
/// Property key for the bounding-box maximum longitude.
pub const PROP_MAX_LNG: &str = "max_lng";
/// Property key for the bounding-box maximum latitude.
pub const PROP_MAX_LAT: &str = "max_lat";
/// Property key for the full geometry (GeoJSON-serialized).
pub const PROP_GEOMETRY: &str = "geometry";

// ============================================================================
// Distance
// ============================================================================

/// Approximate meters per degree of latitude (mean value used by `geo`).
pub const METERS_PER_DEGREE_LAT: f64 = 111_132.0;

/// Mean Earth radius in meters (IUGG recommendation, used by `geo`).
pub const EARTH_RADIUS_METERS: f64 = 6_371_088.0;

/// Great-circle distance between two WGS84 points, in meters (haversine).
pub fn haversine_meters(a: &Point<f64>, b: &Point<f64>) -> f64 {
    Haversine::distance(*a, *b)
}

/// Smallest axis-aligned bounding box (in lon/lat) that encloses every point
/// within `radius_meters` of `center` (great-circle).
///
/// The longitude span widens toward the poles by `1 / cos(latitude)`. The
/// result is exact at the equator and a slight over-estimate elsewhere —
/// ideal for pre-filtering (false positives are removed by the exact
/// predicates downstream).
pub fn bounding_box_for_radius(center: &Point<f64>, radius_meters: f64) -> Rect<f64> {
    let lat_span = radius_meters / METERS_PER_DEGREE_LAT;
    let lng_span = lat_span / center.y().to_radians().cos().abs().max(1e-3);
    Rect::new(
        Coord {
            x: center.x() - lng_span,
            y: center.y() - lat_span,
        },
        Coord {
            x: center.x() + lng_span,
            y: center.y() + lat_span,
        },
    )
}

// ============================================================================
// Bounding boxes
// ============================================================================

/// Bounding box of a geometry, or `None` for an empty geometry.
pub fn geometry_bounds(g: &Geometry<f64>) -> Option<Rect<f64>> {
    g.bounding_rect()
}

/// True when the point lies inside the rectangle (boundary inclusive).
pub fn point_in_rect(p: &Point<f64>, rect: &Rect<f64>) -> bool {
    p.x() >= rect.min().x && p.x() <= rect.max().x && p.y() >= rect.min().y && p.y() <= rect.max().y
}

/// True when two axis-aligned rectangles overlap or touch.
pub fn rects_intersect(a: &Rect<f64>, b: &Rect<f64>) -> bool {
    a.min().x <= b.max().x
        && a.max().x >= b.min().x
        && a.min().y <= b.max().y
        && a.max().y >= b.min().y
}

// ============================================================================
// Containment & intersection predicates
// ============================================================================

/// True when `p` is inside `g` or on its boundary (closed-set semantics).
///
/// Uses the point's [`geo::coordinate_position::CoordPos`] so polygon
/// boundaries, line/linestring traces and hole interiors behave intuitively:
/// points inside polygon holes are **not** contained, while a point lying
/// exactly on a polygon ring or on a linestring **is** contained.
pub fn geometry_contains_point(g: &Geometry<f64>, p: &Point<f64>) -> bool {
    use geo::coordinate_position::{CoordPos, CoordinatePosition};
    !matches!(g.coordinate_position(&p.0), CoordPos::Outside)
}

/// True when geometry `a` fully contains geometry `b` (every point of `b`
/// lies inside or on the boundary of `a`).
pub fn geometry_contains(a: &Geometry<f64>, b: &Geometry<f64>) -> bool {
    a.contains(b)
}

/// True when geometry `a` and geometry `b` share any interior or boundary
/// point. The predicate is symmetric.
pub fn geometries_intersect(a: &Geometry<f64>, b: &Geometry<f64>) -> bool {
    // Fast reject: disjoint bounding boxes cannot intersect.
    if let (Some(ra), Some(rb)) = (a.bounding_rect(), b.bounding_rect()) {
        if !rects_intersect(&ra, &rb) {
            return false;
        }
    }

    // A vertex of either geometry inside the other (interior or boundary)
    // implies overlap. This also covers Point-in-Polygon and Point-on-Line.
    for c in a.coords_iter() {
        if geometry_contains_point(b, &Point::from(c)) {
            return true;
        }
    }
    for c in b.coords_iter() {
        if geometry_contains_point(a, &Point::from(c)) {
            return true;
        }
    }

    // No containment, but their boundary segments may still cross.
    for (a1, a2) in segments_of(a) {
        for (b1, b2) in segments_of(b) {
            if geo::Line::new(a1, a2).intersects(&geo::Line::new(b1, b2)) {
                return true;
            }
        }
    }

    false
}

/// Collect the boundary segments of a geometry as `(start, end)` point pairs.
///
/// Polygon rings (exterior and holes) and multi-part geometries contribute
/// their segments; points contribute none. Used by [`geometries_intersect`]
/// for edge-crossing tests.
fn segments_of(g: &Geometry<f64>) -> Vec<(Point<f64>, Point<f64>)> {
    fn push_line(l: &geo::Line<f64>, out: &mut Vec<(Point<f64>, Point<f64>)>) {
        out.push((Point::from(l.start), Point::from(l.end)));
    }
    fn push_lines<I>(lines: I, out: &mut Vec<(Point<f64>, Point<f64>)>)
    where
        I: Iterator<Item = geo::Line<f64>>,
    {
        for l in lines {
            push_line(&l, out);
        }
    }

    let mut out: Vec<(Point<f64>, Point<f64>)> = Vec::new();
    match g {
        Geometry::Line(l) => push_line(l, &mut out),
        Geometry::LineString(ls) => push_lines(ls.lines_iter(), &mut out),
        Geometry::MultiLineString(mls) => {
            for ls in mls.iter() {
                push_lines(ls.lines_iter(), &mut out);
            }
        }
        Geometry::Polygon(p) => push_lines(p.lines_iter(), &mut out),
        Geometry::MultiPolygon(mp) => {
            for p in mp.iter() {
                push_lines(p.lines_iter(), &mut out);
            }
        }
        Geometry::Rect(r) => push_lines(r.lines_iter(), &mut out),
        Geometry::Triangle(t) => push_lines(t.lines_iter(), &mut out),
        Geometry::GeometryCollection(gc) => {
            for member in gc.iter() {
                out.extend(segments_of(member));
            }
        }
        // Points have no segments.
        Geometry::Point(_) | Geometry::MultiPoint(_) => {}
    }
    out
}

// ============================================================================
// Distance to geometry
// ============================================================================

/// Minimum great-circle distance from `p` to any part of `g`, in meters.
///
/// Points inside the geometry (e.g. inside a polygon, ignoring holes) are at
/// distance `0`. The boundary distance is computed as the haversine distance
/// to the geometry's closest vertex/edge — exact for points and a close
/// approximation for line/polygon boundaries.
pub fn distance_point_to_geometry(p: &Point<f64>, g: &Geometry<f64>) -> f64 {
    if g.contains(p) {
        return 0.0;
    }
    match g.closest_point(p) {
        Closest::Intersection(pt) | Closest::SinglePoint(pt) => haversine_meters(p, &pt),
        // Empty / degenerate geometry: no finite distance.
        Closest::Indeterminate => f64::INFINITY,
    }
}

/// Approximate minimum great-circle distance between two geometries, in meters.
///
/// Returns `0` when the geometries touch or intersect. Otherwise every vertex
/// of each geometry is projected onto the other geometry (an exact
/// point-to-line/polygon closest-point computation) and the minimum haversine
/// distance is returned. This is exact when one geometry is a point, and a
/// close approximation for line/polygon pairs that do not intersect.
pub fn distance_between_geometries(a: &Geometry<f64>, b: &Geometry<f64>) -> f64 {
    if geometries_intersect(a, b) {
        return 0.0;
    }
    let mut best = f64::INFINITY;
    for c in a.coords_iter() {
        let d = distance_point_to_geometry(&Point::from(c), b);
        if d < best {
            best = d;
        }
    }
    for c in b.coords_iter() {
        let d = distance_point_to_geometry(&Point::from(c), a);
        if d < best {
            best = d;
        }
    }
    best
}

// ============================================================================
// Tile projection (Web Mercator / slippy-map)
// ============================================================================

/// The maximum latitude representable in Web Mercator (EPSG:3857), in degrees.
/// Beyond this the projection diverges, so latitude is clamped to it.
pub const MAX_MERCATOR_LATITUDE: f64 = 85.051_128_779_806_59;

/// The axis-aligned lon/lat bounding box of a slippy-map tile `z`/`x`/`y`.
///
/// Uses the standard Web Mercator (EPSG:3857) tile scheme: at zoom `z` the
/// world is a `2^z` by `2^z` grid, `x` is the column (0 at lon -180) and `y`
/// is the row (0 at the top). The returned `Rect` is in WGS84 degrees and is
/// the exact window to pass to a `SpatialQuery::BoundingBox`/`Intersects`.
pub fn tile_bounds(z: u8, x: u32, y: u32) -> Rect<f64> {
    let n = 2f64.powi(z as i32);
    let x = x as f64;
    let y = y as f64;
    let min_lon = x / n * 360.0 - 180.0;
    let max_lon = (x + 1.0) / n * 360.0 - 180.0;
    // Mercator rows grow southward: y = 0 is the top (maximum latitude).
    let max_lat = world_y_to_lat(y / n);
    let min_lat = world_y_to_lat((y + 1.0) / n);
    Rect::new(
        Coord {
            x: min_lon,
            y: min_lat,
        },
        Coord {
            x: max_lon,
            y: max_lat,
        },
    )
}

/// The tile `(x, y)` containing `p` at zoom `z`.
///
/// Longitude is clamped to +/-180 and latitude to +/-MAX_MERCATOR_LATITUDE, so
/// points slightly outside the valid Mercator range still map to a tile.
pub fn point_to_tile(p: &Point<f64>, z: u8) -> (u32, u32) {
    let n = 2f64.powi(z as i32);
    let max = (n as i64) - 1;
    let lon = p.x().clamp(-180.0, 180.0);
    let lat = p.y().clamp(-MAX_MERCATOR_LATITUDE, MAX_MERCATOR_LATITUDE);
    let tx = ((lon + 180.0) / 360.0 * n).floor() as i64;
    let ty = lat_to_world_y(lat, n).floor() as i64;
    (tx.clamp(0, max) as u32, ty.clamp(0, max) as u32)
}

/// Convert a normalized world `y` (0..1, top-to-bottom) to latitude in degrees.
fn world_y_to_lat(world_y: f64) -> f64 {
    (std::f64::consts::PI * (1.0 - 2.0 * world_y))
        .sinh()
        .atan()
        .to_degrees()
}

/// Convert latitude in degrees to a world `y` (0..`n`) at `n` tiles per side.
fn lat_to_world_y(lat: f64, n: f64) -> f64 {
    let lat = lat.to_radians();
    let mercator = (std::f64::consts::FRAC_PI_4 + lat / 2.0).tan().ln();
    (1.0 - mercator / std::f64::consts::PI) / 2.0 * n
}

/// Project a WGS84 longitude/latitude into tile-local coordinates for the
/// slippy-map tile `z`/`x`/`y`, scaled to `extent` (vector tiles use 4096).
///
/// Points inside the tile map to `0..=extent`; features that merely cross the
/// tile fall outside that range (clip or clamp them downstream).
pub fn lonlat_to_tile_coord(lon: f64, lat: f64, z: u8, x: u32, y: u32, extent: f64) -> (f64, f64) {
    let n = 2f64.powi(z as i32);
    let lat = lat.clamp(-MAX_MERCATOR_LATITUDE, MAX_MERCATOR_LATITUDE);
    let world_x = (lon + 180.0) / 360.0 * n;
    let world_y = lat_to_world_y(lat, n);
    ((world_x - x as f64) * extent, (world_y - y as f64) * extent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{LineString, Polygon};

    fn poly_from(coords: &[(f64, f64)]) -> Polygon<f64> {
        Polygon::new(LineString::from(coords.to_vec()), Vec::new())
    }

    #[test]
    fn haversine_matches_reference_values() {
        // New York City → London (from geo's own docs).
        let nyc = Point::new(-74.006, 40.7128);
        let london = Point::new(-0.1278, 51.5074);
        let d = haversine_meters(&nyc, &london);
        assert!((d - 5_570_230.0).abs() < 50_000.0, "got {d}");

        // Zero distance for identical points.
        assert_eq!(haversine_meters(&nyc, &nyc), 0.0);

        // ~1° of latitude ≈ 111 km.
        let d = haversine_meters(&Point::new(0.0, 0.0), &Point::new(0.0, 1.0));
        assert!((d - 111_132.0).abs() < 1_000.0, "got {d}");
    }

    #[test]
    fn radius_bounding_box_covers_exact_distance() {
        // On the equator, a 111.132 km radius spans ±1° lat and ±1° lng.
        let center = Point::new(0.0, 0.0);
        let rect = bounding_box_for_radius(&center, 111_132.0);
        assert!((rect.min().x + 1.0).abs() < 1e-6);
        assert!((rect.max().x - 1.0).abs() < 1e-6);
        assert!((rect.min().y + 1.0).abs() < 1e-6);
        assert!((rect.max().y - 1.0).abs() < 1e-6);

        // Any point within the radius lies inside the box (no false negatives).
        let near = Point::new(0.5, 0.5); // ≈78 km from origin
        assert!(point_in_rect(&near, &rect));
        assert!(haversine_meters(&center, &near) < 111_132.0);
    }

    #[test]
    fn rect_predicates() {
        let r = Rect::new(Coord { x: 0.0, y: 0.0 }, Coord { x: 10.0, y: 10.0 });
        assert!(point_in_rect(&Point::new(5.0, 5.0), &r));
        assert!(point_in_rect(&Point::new(0.0, 10.0), &r)); // boundary inclusive
        assert!(!point_in_rect(&Point::new(10.1, 5.0), &r));

        let overlapping = Rect::new(Coord { x: 5.0, y: 5.0 }, Coord { x: 15.0, y: 15.0 });
        let touching = Rect::new(Coord { x: 10.0, y: 0.0 }, Coord { x: 20.0, y: 5.0 });
        let disjoint = Rect::new(Coord { x: 100.0, y: 100.0 }, Coord { x: 110.0, y: 110.0 });
        assert!(rects_intersect(&r, &overlapping));
        assert!(rects_intersect(&r, &touching));
        assert!(!rects_intersect(&r, &disjoint));
    }

    #[test]
    fn geometry_contains_point_handles_holes_and_boundaries() {
        let exterior = LineString::from(vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]);
        let hole = LineString::from(vec![
            (4.0, 4.0),
            (6.0, 4.0),
            (6.0, 6.0),
            (4.0, 6.0),
            (4.0, 4.0),
        ]);
        let poly = Geometry::Polygon(Polygon::new(exterior, vec![hole]));

        assert!(geometry_contains_point(&poly, &Point::new(1.0, 1.0)));
        assert!(!geometry_contains_point(&poly, &Point::new(5.0, 5.0))); // inside the hole
        assert!(!geometry_contains_point(&poly, &Point::new(20.0, 20.0)));
        assert!(geometry_contains_point(&poly, &Point::new(0.0, 5.0))); // boundary inclusive

        // A point lying exactly on a linestring is "contained" by the line.
        let line = Geometry::LineString(LineString::from(vec![(0.0, 0.0), (10.0, 10.0)]));
        assert!(geometry_contains_point(&line, &Point::new(5.0, 5.0)));
        assert!(!geometry_contains_point(&line, &Point::new(5.0, 6.0)));
    }

    #[test]
    fn geometry_contains_full_geometry() {
        let outer = Geometry::Polygon(poly_from(&[
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 20.0),
            (0.0, 20.0),
            (0.0, 0.0),
        ]));
        let inner = Geometry::Polygon(poly_from(&[
            (5.0, 5.0),
            (10.0, 5.0),
            (10.0, 10.0),
            (5.0, 10.0),
            (5.0, 5.0),
        ]));
        let overlapping = Geometry::Polygon(poly_from(&[
            (15.0, 15.0),
            (25.0, 15.0),
            (25.0, 25.0),
            (15.0, 25.0),
            (15.0, 15.0),
        ]));

        assert!(geometry_contains(&outer, &inner));
        assert!(!geometry_contains(&outer, &overlapping));
        assert!(!geometry_contains(&inner, &outer));
        assert!(geometry_contains(&inner, &inner));
    }

    #[test]
    fn geometries_intersect_predicate() {
        let a = Geometry::Polygon(poly_from(&[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]));
        let overlapping = Geometry::Polygon(poly_from(&[
            (5.0, 5.0),
            (15.0, 5.0),
            (15.0, 15.0),
            (5.0, 15.0),
            (5.0, 5.0),
        ]));
        let disjoint = Geometry::Polygon(poly_from(&[
            (50.0, 50.0),
            (60.0, 50.0),
            (60.0, 60.0),
            (50.0, 60.0),
            (50.0, 50.0),
        ]));
        // A polygon nested inside another still intersects it.
        let nested = Geometry::Polygon(poly_from(&[
            (2.0, 2.0),
            (3.0, 2.0),
            (3.0, 3.0),
            (2.0, 3.0),
            (2.0, 2.0),
        ]));

        assert!(geometries_intersect(&a, &overlapping));
        assert!(!geometries_intersect(&a, &disjoint));
        assert!(geometries_intersect(&a, &nested));
        // A point inside the polygon intersects it.
        let pt = Geometry::Point(Point::new(1.0, 1.0));
        assert!(geometries_intersect(&a, &pt));
        // A point on the boundary intersects it.
        let edge_pt = Geometry::Point(Point::new(0.0, 5.0));
        assert!(geometries_intersect(&a, &edge_pt));
    }

    #[test]
    fn distance_to_geometry_values() {
        let poly = Geometry::Polygon(poly_from(&[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]));

        // Inside → zero.
        assert_eq!(
            distance_point_to_geometry(&Point::new(5.0, 5.0), &poly),
            0.0
        );

        // ~10° lat north of the top edge → ≈ 1.11e6 m.
        let d = distance_point_to_geometry(&Point::new(5.0, 20.0), &poly);
        assert!(d > 1.0e6 && d < 1.2e6, "got {d}");

        // Distance to a point geometry is plain haversine.
        let other = Geometry::Point(Point::new(0.0, 1.0));
        let d = distance_point_to_geometry(&Point::new(0.0, 0.0), &other);
        assert!((d - 111_132.0).abs() < 1_000.0, "got {d}");
    }

    #[test]
    fn tile_bounds_cover_the_world_at_zoom_zero() {
        let bounds = tile_bounds(0, 0, 0);
        assert!((bounds.min().x + 180.0).abs() < 1e-9);
        assert!((bounds.max().x - 180.0).abs() < 1e-9);
        assert!((bounds.max().y - MAX_MERCATOR_LATITUDE).abs() < 1e-6);
        assert!((bounds.min().y + MAX_MERCATOR_LATITUDE).abs() < 1e-6);
    }

    #[test]
    fn tile_bounds_match_slippy_map_quadrants() {
        let top_left = tile_bounds(1, 0, 0);
        assert!((top_left.min().x + 180.0).abs() < 1e-9);
        assert!(top_left.max().x.abs() < 1e-9);
        assert!(top_left.min().y.abs() < 1e-9);
        assert!((top_left.max().y - MAX_MERCATOR_LATITUDE).abs() < 1e-6);

        let bottom_right = tile_bounds(1, 1, 1);
        assert!(bottom_right.min().x.abs() < 1e-9);
        assert!((bottom_right.max().x - 180.0).abs() < 1e-9);
        assert!((bottom_right.min().y + MAX_MERCATOR_LATITUDE).abs() < 1e-6);
        assert!(bottom_right.max().y.abs() < 1e-9);
    }

    #[test]
    fn point_to_tile_maps_singapore_and_round_trips() {
        let singapore = Point::new(103.85, 1.35);
        let (x, y) = point_to_tile(&singapore, 10);
        assert_eq!((x, y), (807, 508));

        let bounds = tile_bounds(10, x, y);
        assert!(point_in_rect(&singapore, &bounds));
    }

    #[test]
    fn point_to_tile_clamps_poles_and_antimeridian() {
        assert_eq!(point_to_tile(&Point::new(180.0, 0.0), 2), (3, 2));
        assert_eq!(point_to_tile(&Point::new(0.0, 90.0), 2), (2, 0));
        assert_eq!(point_to_tile(&Point::new(0.0, -90.0), 2), (2, 3));
    }

    #[test]
    fn lonlat_to_tile_coord_maps_corners_and_interior() {
        let (z, x, y) = (10u8, 807u32, 508u32);
        let bounds = tile_bounds(z, x, y);
        let extent = 4096.0;

        // Top-left corner of the tile maps to ~(0, 0); bottom-right to ~(extent, extent).
        let tl = lonlat_to_tile_coord(bounds.min().x, bounds.max().y, z, x, y, extent);
        let br = lonlat_to_tile_coord(bounds.max().x, bounds.min().y, z, x, y, extent);
        assert!(tl.0.abs() < 1e-6 && tl.1.abs() < 1e-6, "top-left {tl:?}");
        assert!(
            (br.0 - extent).abs() < 1e-6 && (br.1 - extent).abs() < 1e-6,
            "bottom-right {br:?}"
        );

        // An interior point lands strictly inside 0..extent.
        let (cx, cy) = lonlat_to_tile_coord(103.85, 1.35, z, x, y, extent);
        assert!(
            cx > 0.0 && cx < extent && cy > 0.0 && cy < extent,
            "center {cx},{cy}"
        );
    }
}

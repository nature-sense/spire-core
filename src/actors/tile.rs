// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! TileActor — slippy-map tile features for map UIs, backed by the memory graph.
//!
//! Map UIs consume vector tiles as `z`/`x`/`y` quadrants of the Web Mercator
//! (EPSG:3857) world. This actor answers "which features intersect this tile?"
//! by translating the tile to a WGS84 bounding box (`crate::spatial::tile_bounds`)
//! and asking `MemoryGraphActor` with a `SpatialQuery::Intersects`. Results are
//! cached per `(filters, z, x, y)` so panning a viewport never re-queries the
//! store for tiles it already has.
//!
//! Encoding the returned features into MVT bytes is a separate, pure step (a
//! features -> MVT encoder) that callers compose on top of
//! [`TileMessage::GetTileFeatures`]. The graph actor is untouched: tile serving
//! is a read-only consumer of the same `SpatialQuery` API as any other caller.

use anyhow::Result;
use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};
use tokio::sync::{mpsc, oneshot};

use crate::actors::Actor;
use crate::models::memory_graph::{AttrNode, SpatialQuery};
use crate::subsystems::graph::memory_graph::MemoryGraphMessage;

/// Filters narrowing which features a tile contains.
///
/// An empty filter (all `None`) returns every feature intersecting the tile.
#[derive(Debug, Clone, Default)]
pub struct TileFilters {
    pub node_type: Option<String>,
    pub subtype: Option<String>,
    pub limit: Option<usize>,
}

impl TileFilters {
    /// Canonical string used as the filter half of the tile cache key.
    pub fn cache_key(&self) -> String {
        format!(
            "{}/{}",
            self.node_type.as_deref().unwrap_or("*"),
            self.subtype.as_deref().unwrap_or("*")
        )
    }
}

/// Messages for [`TileActor`].
#[derive(Debug)]
pub enum TileMessage {
    /// All features intersecting slippy-map tile `z`/`x`/`y` (optionally
    /// filtered by [`TileFilters`]). Repeated requests hit the LRU cache.
    GetTileFeatures {
        filters: TileFilters,
        z: u8,
        x: u32,
        y: u32,
        reply_to: oneshot::Sender<Result<Vec<AttrNode>>>,
    },
    /// MVT-encoded bytes for slippy-map tile `z`/`x`/`y`. Features come from
    /// the same cached path as [`TileMessage::GetTileFeatures`] and are
    /// encoded with [`crate::tiles::encode_tile`].
    GetTile {
        filters: TileFilters,
        z: u8,
        x: u32,
        y: u32,
        reply_to: oneshot::Sender<Result<Vec<u8>>>,
    },
    /// Drop every cached tile.
    ClearCache { reply_to: oneshot::Sender<()> },
}

/// LRU cache key: the filter key plus the tile coordinate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TileKey {
    filters: String,
    z: u8,
    x: u32,
    y: u32,
}

/// A tiny bounded LRU mapping tile -> feature sets.
struct TileCache {
    capacity: usize,
    map: HashMap<TileKey, Vec<AttrNode>>,
    order: VecDeque<TileKey>,
}

impl TileCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, key: &TileKey) -> Option<Vec<AttrNode>> {
        if !self.map.contains_key(key) {
            return None;
        }
        // Move to the back = most recently used.
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            if let Some(k) = self.order.remove(pos) {
                self.order.push_back(k);
            }
        }
        self.map.get(key).cloned()
    }

    fn put(&mut self, key: TileKey, value: Vec<AttrNode>) {
        // Update in place when present; otherwise evict the LRU entry first.
        if let Some(existing) = self.map.get_mut(&key) {
            *existing = value;
            return;
        }
        if self.map.len() >= self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.map.remove(&evicted);
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }

    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}

/// Default number of tiles held in the LRU cache.
const DEFAULT_TILE_CACHE_CAPACITY: usize = 512;

/// Serves tile feature sets from the memory graph, caching per tile.
///
/// Spawn like any actor: `TileActor::new(mg_tx).spawn(rx)`. Holds a sender to
/// `MemoryGraphActor` only — it never owns graph state.
pub struct TileActor {
    memory_graph_tx: mpsc::Sender<MemoryGraphMessage>,
    cache: TileCache,
}

impl TileActor {
    pub fn new(memory_graph_tx: mpsc::Sender<MemoryGraphMessage>) -> Self {
        Self {
            memory_graph_tx,
            cache: TileCache::new(DEFAULT_TILE_CACHE_CAPACITY),
        }
    }

    /// Override the LRU cache capacity (default 512 tiles).
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.cache = TileCache::new(capacity);
        self
    }

    async fn get_tile_features(
        &mut self,
        filters: &TileFilters,
        z: u8,
        x: u32,
        y: u32,
    ) -> Result<Vec<AttrNode>> {
        let key = TileKey {
            filters: filters.cache_key(),
            z,
            x,
            y,
        };
        if let Some(hit) = self.cache.get(&key) {
            return Ok(hit);
        }

        let bounds = crate::spatial::tile_bounds(z, x, y);
        let (tx, rx) = oneshot::channel();
        self.memory_graph_tx
            .send(MemoryGraphMessage::SpatialQuery {
                query: SpatialQuery::Intersects {
                    geometry: geo::Geometry::Rect(bounds),
                },
                node_type: filters.node_type.clone(),
                subtype: filters.subtype.clone(),
                limit: filters.limit,
                reply_to: tx,
            })
            .await
            .map_err(|e| anyhow::anyhow!("send spatial query failed: {e}"))?;
        let result = rx
            .await
            .map_err(|e| anyhow::anyhow!("spatial reply dropped: {e}"))??;

        let nodes: Vec<AttrNode> = result.nodes.into_iter().map(|scored| scored.node).collect();
        self.cache.put(key, nodes.clone());
        Ok(nodes)
    }

    /// MVT bytes for a tile: the cached feature set encoded by `crate::tiles`.
    /// The CPU-bound encode step runs on a blocking task, off the mailbox.
    async fn get_tile(&mut self, filters: &TileFilters, z: u8, x: u32, y: u32) -> Result<Vec<u8>> {
        let features = self.get_tile_features(filters, z, x, y).await?;
        tokio::task::spawn_blocking(move || crate::tiles::encode_tile(&features, z, x, y))
            .await
            .map_err(|e| anyhow::anyhow!("tile encode task failed: {e}"))?
    }
}

#[async_trait]
impl Actor for TileActor {
    type Message = TileMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            TileMessage::GetTileFeatures {
                filters,
                z,
                x,
                y,
                reply_to,
            } => {
                let result = self.get_tile_features(&filters, z, x, y).await;
                let _ = reply_to.send(result);
            }
            TileMessage::GetTile {
                filters,
                z,
                x,
                y,
                reply_to,
            } => {
                let result = self.get_tile(&filters, z, x, y).await;
                let _ = reply_to.send(result);
            }
            TileMessage::ClearCache { reply_to } => {
                self.cache.clear();
                let _ = reply_to.send(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(z: u8, x: u32, y: u32) -> TileKey {
        TileKey {
            filters: "*".to_string(),
            z,
            x,
            y,
        }
    }

    fn node(id: &str) -> AttrNode {
        AttrNode {
            id: id.to_string(),
            node_type: "Sensor".to_string(),
            subtype: None,
            name: id.to_string(),
            description: None,
            properties: Default::default(),
            embedding_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: 1,
        }
    }

    #[test]
    fn lru_evicts_least_recently_used() {
        let mut cache = TileCache::new(2);
        cache.put(key(10, 0, 0), vec![node("a")]);
        cache.put(key(10, 0, 1), vec![node("b")]);
        // Touch the first entry so the second becomes least-recently-used.
        assert_eq!(cache.get(&key(10, 0, 0)).unwrap()[0].id(), "a");
        cache.put(key(10, 1, 0), vec![node("c")]);
        assert!(cache.get(&key(10, 0, 1)).is_none(), "LRU entry evicted");
        assert_eq!(cache.get(&key(10, 0, 0)).unwrap()[0].id(), "a");
        assert_eq!(cache.get(&key(10, 1, 0)).unwrap()[0].id(), "c");
    }

    #[test]
    fn filters_make_distinct_cache_keys() {
        let plain = TileFilters::default();
        let typed = TileFilters {
            node_type: Some("Sensor".to_string()),
            ..Default::default()
        };
        assert_ne!(plain.cache_key(), typed.cache_key());
        assert_eq!(plain.cache_key(), "*/*");
        assert_eq!(typed.cache_key(), "Sensor/*");
    }
}

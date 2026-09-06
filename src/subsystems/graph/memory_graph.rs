// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! MemoryGraphActor — backed by SeleneDB's `GraphDb` with GQL persistence.
//!
//! This actor is the sole data store for the system, owning graph nodes, edges,
//! and vector embeddings. All storage is delegated to `GraphDb` (SeleneDB),
//! which provides lock-free reads, serialized writes, and optional WAL persistence.
//!
//! All data access is through GQL statements via `execute_gql_write` and
//! `execute_gql_query`. No low-level SharedGraph API is used.
//!
//! # ID Mapping
//!
//! The external API uses UUID-based `String` IDs (for compatibility with the
//! TypeScript extension), while SeleneDB uses compact `u64` IDs (`NodeId`/`EdgeId`).
//! This actor maintains a bidirectional mapping between the two, stored as
//! properties on the nodes/edges themselves.

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;

use selene_db_core::value::Value;
use selene_db_core::vector::VectorMetric;

use spire_actor::Actor;

use crate::graph::GraphDb;
use crate::models::embedding::Embedder;

/// Canonical cross-compilation platform definitions cross the crate boundary
/// as generic JSON (`{ "id", "name", "properties": {flat map} }`); spire-core
/// owns the typed `Platform` YAML schema + view.
use crate::models::memory_graph::{
    AttrNode, ContextSearchResult, DistanceScoredNode, GraphEdge, McpConfigFile,
    McpServerConfigEntry, MemoryEntry, MemoryMetadata, NodeUpdate, ProjectSnapshot, ProjectStats,
    RelationshipInput, RelationshipType, RetrievalSource, ScoredNode, SearchOptions, SpatialQuery,
    SpatialQueryResult, StreamOp, StreamOpResult, TransactionRequest, TraversalDirection,
    TraversalOptions, TraversalPath, TraversalResult,
};

// ============================================================================
// GQL Schema Constants
// ============================================================================

/// Label used for all Spire graph nodes in SeleneDB.
const LABEL_SPIRE_NODE: &str = "SpireNode";
/// Label used for config key-value storage nodes.
const LABEL_CONFIG: &str = "SpireConfig";
/// Property key for the UUID string.
const PROP_UUID: &str = "uuid";
/// Property key for the node type.
const PROP_NODE_TYPE: &str = "node_type";
/// Property key for the node subtype.
const PROP_SUBTYPE: &str = "subtype";
/// Property key for the node name.
const PROP_NAME: &str = "name";
/// Property key for the node description.
const PROP_DESCRIPTION: &str = "description";
/// Property key for the embedding ID.
const PROP_EMBEDDING_ID: &str = "embedding_id";
/// Property key for the created_at timestamp.
const PROP_CREATED_AT: &str = "created_at";
/// Property key for the updated_at timestamp.
const PROP_UPDATED_AT: &str = "updated_at";
/// Property key for the version number.
const PROP_VERSION: &str = "version";
/// Property key for the edge type.
const PROP_EDGE_TYPE: &str = "edge_type";
/// Property key for the edge weight.
const PROP_WEIGHT: &str = "weight";
/// Property key for the config value.
const PROP_CONFIG_VALUE: &str = "config_value";

/// Convert a `RelationshipType` to a valid GQL edge label string.
///
/// Unit variants (e.g. `HasDecision`) are serialized via serde and trimmed of
/// quotes, producing a valid label like `has_decision`. The `Custom(String)`
/// variant returns the inner string directly, avoiding serde's JSON object
/// serialization (`{"Custom":"HAS_BUILD_SYSTEM"}`) which would produce an
/// invalid GQL label atom.
fn relationship_type_to_gql_label(rel_type: &RelationshipType) -> String {
    match rel_type {
        RelationshipType::Custom(s) => s.clone(),
        _ => serde_json::to_string(rel_type)
            .unwrap_or_else(|_| format!("{:?}", rel_type))
            .trim_matches('"')
            .to_string(),
    }
}

// ============================================================================

// MemoryGraphMessage Enum — 14 variants matching IMemoryGraph API
// ============================================================================

/// Messages for the MemoryGraph actor.
///
/// This actor is the sole data store for the system, owning graph nodes, edges,
/// and vector embeddings directly (no separate GraphActor or VectorActor).
pub enum MemoryGraphMessage {
    // ── Lifecycle ────────────────────────────────────────
    /// Initialize the graph database with the given data directory.
    /// Creates the GraphDb instance and rebuilds the UUID cache.
    Initialize {
        data_dir: PathBuf,
        reply_to: tokio::sync::oneshot::Sender<Result<()>>,
    },
    /// Initialize the embedding model.
    /// If `embedder` is provided, use it directly (avoids creating a second
    /// CandleEmbedder instance, which does blocking I/O).
    /// If `embedder` is None, creates a new one via `crate::embedder::create_embedder()`.
    InitializeEmbedder {
        model_path: Option<PathBuf>,
        embedder: Option<Arc<dyn Embedder>>,
        reply_to: tokio::sync::oneshot::Sender<Result<()>>,
    },

    // ── Node Operations ─────────────────────────────────
    /// Store a node from the open `AttrNode` envelope. The arbitrary
    /// `node_type` string is written verbatim (no enum mapping), scalar
    /// properties inline, complex (array/object) properties SET individually.
    StoreAttrNode {
        node: AttrNode,
        reply_to: tokio::sync::oneshot::Sender<Result<AttrNode>>,
    },
    /// Upsert an `AttrNode`: reuse the existing UUID when a node with the same
    /// (node_type, subtype, name) already exists so relationships stay valid.
    MergeAttrNode {
        node: AttrNode,
        reply_to: tokio::sync::oneshot::Sender<Result<AttrNode>>,
    },
    /// Read a node back as the open `AttrNode` envelope, preserving the
    /// arbitrary `node_type` discriminator.
    GetAttrNode {
        id: String,
        reply_to: tokio::sync::oneshot::Sender<Result<Option<AttrNode>>>,
    },
    /// Query nodes as the open `AttrNode` envelope. Filters on the arbitrary
    /// `node_type` string, subtype and/or name; `limit` caps the result set.
    QueryAttrNodes {
        node_type: Option<String>,
        subtype: Option<String>,
        name: Option<String>,
        limit: Option<u32>,
        reply_to: tokio::sync::oneshot::Sender<Result<Vec<AttrNode>>>,
    },
    UpdateNode {
        id: String,
        updates: NodeUpdate,
        reply_to: tokio::sync::oneshot::Sender<Result<AttrNode>>,
    },
    DeleteNode {
        id: String,
        reply_to: tokio::sync::oneshot::Sender<Result<()>>,
    },

    // ── Relationship Operations ──────────────────────────
    CreateRelationship {
        rel: RelationshipInput,
        reply_to: tokio::sync::oneshot::Sender<Result<GraphEdge>>,
    },
    GetRelationships {
        node_id: String,
        reply_to: tokio::sync::oneshot::Sender<Result<Vec<GraphEdge>>>,
    },
    DeleteRelationship {
        id: String,
        reply_to: tokio::sync::oneshot::Sender<Result<()>>,
    },

    // ── Traversal ────────────────────────────────────────
    Traverse {
        start_node_id: String,
        options: TraversalOptions,
        reply_to: tokio::sync::oneshot::Sender<Result<TraversalResult>>,
    },

    // ── Context & Memory ─────────────────────────────────
    GetProjectContext {
        reply_to: tokio::sync::oneshot::Sender<Result<ProjectSnapshot>>,
    },
    SearchContext {
        query: String,
        options: Option<SearchOptions>,
        reply_to: tokio::sync::oneshot::Sender<Result<ContextSearchResult>>,
    },
    AddMemory {
        text: String,
        metadata: Option<MemoryMetadata>,
        reply_to: tokio::sync::oneshot::Sender<Result<MemoryEntry>>,
    },
    Recall {
        query: String,
        limit: Option<usize>,
        reply_to: tokio::sync::oneshot::Sender<Result<Vec<MemoryEntry>>>,
    },

    // ── Spatial Queries ─────────────────────────────────
    /// Run a spatial predicate over nodes that carry spatial properties
    /// (`latitude`/`longitude` points, or `min_lng`/`min_lat`/`max_lng`/
    /// `max_lat` bounding boxes with optional `geometry`). See
    /// `crate::spatial` for the coordinate convention.
    SpatialQuery {
        query: SpatialQuery,
        node_type: Option<String>,
        subtype: Option<String>,
        limit: Option<usize>,
        reply_to: tokio::sync::oneshot::Sender<Result<SpatialQueryResult>>,
    },

    // ── Config Storage ───────────────────────────────────
    SetConfig {
        key: String,
        value: serde_json::Value,
        reply_to: tokio::sync::oneshot::Sender<Result<()>>,
    },
    GetConfig {
        key: String,
        reply_to: tokio::sync::oneshot::Sender<Result<Option<serde_json::Value>>>,
    },

    // ── MCP Config Storage ───────────────────────────────
    /// Bootstrap MCP server config from a JSON file into the graph.
    /// Reads the file, parses it, deletes any existing SpireMcpConfig nodes,
    /// and stores each server as a new SpireMcpConfig node (sync semantics).
    BootstrapMcpConfig {
        config_path: PathBuf,
        reply_to: tokio::sync::oneshot::Sender<Result<()>>,
    },
    /// Get all MCP server config entries from the graph.
    GetMcpConfig {
        reply_to: tokio::sync::oneshot::Sender<Result<Vec<McpServerConfigEntry>>>,
    },

    // ── Platform Config Storage ─────────────────────────
    /// Seed the graph with platform definitions from the platform registry.
    /// Each element is `{ "id", "name", "properties": {flat map} }` — the
    /// caller (spire-core) owns the platform YAML schema; the graph stores
    /// the nodes generically.
    BootstrapPlatforms {
        platforms: Vec<serde_json::Value>,
        reply_to: tokio::sync::oneshot::Sender<Result<()>>,
    },
    /// Get all platform definitions from the graph as generic JSON nodes
    /// (`{ "id", "name", "properties": {flat map} }`); spire-core rebuilds its
    /// typed `Platform` view from the properties.
    GetPlatforms {
        reply_to: tokio::sync::oneshot::Sender<Result<Vec<serde_json::Value>>>,
    },

    // ── Batch / Atomic Operations ─────────────────────────
    /// Seed the graph with intent definitions from a JSON file.
    /// Reads the file, parses it, and stores each intent as a SpireNode.
    SeedIntents {
        config_path: PathBuf,
        reply_to: tokio::sync::oneshot::Sender<Result<()>>,
    },
    /// Atomically update a node (delete + insert as one unit).
    /// Safer than the current two-step approach — if the INSERT fails,
    /// the DELETE is rolled back and the node is unchanged.
    AtomicUpdateNode {
        id: String,
        updates: NodeUpdate,
        reply_to: tokio::sync::oneshot::Sender<Result<AttrNode>>,
    },
    /// Execute an atomic batch of arbitrary GQL statements.
    BatchGql {
        statements: Vec<String>,
        reply_to: tokio::sync::oneshot::Sender<Result<()>>,
    },
    /// Open a transaction stream for atomic multi-operation batches.
    ///
    /// Returns an `mpsc::Sender<TransactionRequest>` that the caller can use
    /// to push individual operations. Each operation gets a per-op response
    /// via its embedded `oneshot::Sender`. The stream is closed by sending
    /// `StreamOp::Commit` or `StreamOp::Rollback`, or by dropping the sender.
    /// On sender drop without explicit Commit/Rollback, the transaction is
    /// committed automatically (RAII-style, matching `GraphDbTransaction`).
    OpenTransactionStream {
        reply_to: tokio::sync::oneshot::Sender<mpsc::Sender<TransactionRequest>>,
    },

    // ── Maintenance ──────────────────────────────────────
    Sync {
        reply_to: tokio::sync::oneshot::Sender<Result<()>>,
    },
}

// ============================================================================
// MemoryGraphActor
// ============================================================================

/// The sole data store actor, backed by SeleneDB's `GraphDb`.
///
/// Owns graph nodes, edges, and vector embeddings via `GraphDb`.
/// All metadata is persisted directly in SeleneDB using GQL statements.
/// No separate GraphActor or VectorActor — all operations are handled inline.
///
/// Enforces schema constraints:
/// - Unique `(type, name)` per node
/// - Referential integrity for relationships (from_id / to_id must exist)
/// - Acyclic `depends_on` relationships
pub struct MemoryGraphActor {
    /// The SeleneDB-backed graph database.
    graph_db: Option<Arc<GraphDb>>,

    /// Embedder for text → vector generation.
    embedder: Option<Arc<dyn Embedder>>,

    /// Data directory for snapshot persistence.
    /// Set during `Initialize` and used by `Sync` to write snapshots.
    data_dir: Option<PathBuf>,

    /// Channel sender for debounced snapshot scheduling.
    /// Each write mutation sends a unit value through this channel;
    /// a background task receives it, debounces for 2 seconds, then
    /// writes a snapshot to disk.
    snapshot_tx: Option<mpsc::UnboundedSender<()>>,

    /// Handle to the background snapshot task, so it can be aborted
    /// on drop if needed.
    snapshot_task_handle: Option<tokio::task::JoinHandle<()>>,
}

impl Default for MemoryGraphActor {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryGraphActor {
    pub fn new() -> Self {
        let (snapshot_tx, _snapshot_rx) = mpsc::unbounded_channel::<()>();
        Self {
            graph_db: None,
            embedder: None,
            data_dir: None,
            snapshot_tx: Some(snapshot_tx),
            snapshot_task_handle: None,
        }
    }

    /// Initialize the graph database from a data directory.
    ///
    /// Uses `GraphDb::recover()` to restore from the latest snapshot + WAL,
    /// providing cross-session persistence. If no snapshot exists, this falls
    /// back to an empty WAL-backed graph (equivalent to `new_with_wal`).
    ///
    /// If recovery fails (e.g., WAL/snapshot sequence mismatch after a crash),
    /// this method cleans up stale persistence files and starts fresh with a
    /// new WAL-backed graph. This ensures the system can always start even
    /// after a corrupted or inconsistent persistence state.
    fn init_graph(&mut self, data_dir: &PathBuf) -> Result<()> {
        let graph_id = selene_db_core::identity::GraphId::new(1);

        // Attempt recovery from existing snapshot + WAL
        let graph_db = match GraphDb::recover(data_dir, graph_id) {
            Ok(db) => {
                info!(
                    "MemoryGraph: graph database recovered from: {}",
                    data_dir.display()
                );
                Arc::new(db)
            }
            Err(e) => {
                let msg = e.to_string();
                warn!(
                    "MemoryGraph: recovery failed ({}), attempting clean start",
                    msg
                );

                if msg.contains("wal snapshot sequence")
                    && msg.contains("does not match applied snapshot")
                {
                    warn!("MemoryGraph: WAL/snapshot sequence mismatch detected — removing stale WAL, keeping snapshots");
                    let wal_path = data_dir.join("wal.log");
                    if wal_path.exists() {
                        std::fs::remove_file(&wal_path).map_err(|e| {
                            anyhow::anyhow!("Failed to remove stale WAL file: {}", e)
                        })?;
                        info!("MemoryGraph: removed stale WAL file: {:?}", wal_path);
                    }
                    let spire_wal_path = data_dir.join("spire.wal");
                    if spire_wal_path.exists() {
                        std::fs::remove_file(&spire_wal_path).map_err(|e| {
                            anyhow::anyhow!("Failed to remove stale spire.wal: {}", e)
                        })?;
                        info!("MemoryGraph: removed stale spire.wal: {:?}", spire_wal_path);
                    }
                }

                info!("MemoryGraph: recovering graph from: {} (stale WAL removed, snapshots preserved)", data_dir.display());
                let fresh_db = GraphDb::recover(data_dir, graph_id)
                    .map_err(|e| anyhow::anyhow!("Failed to recover fresh graph: {}", e))?;
                Arc::new(fresh_db)
            }
        };

        self.graph_db = Some(graph_db.clone());
        self.data_dir = Some(data_dir.clone());

        let (new_tx, snapshot_rx) = mpsc::unbounded_channel::<()>();
        self.snapshot_tx = Some(new_tx);
        let handle = Self::spawn_snapshot_task(data_dir.clone(), graph_db, snapshot_rx);
        self.snapshot_task_handle = Some(handle);

        Ok(())
    }

    /// Initialize the embedding model.
    fn init_embedder(
        &mut self,
        _model_path: Option<PathBuf>,
        embedder: Option<Arc<dyn Embedder>>,
    ) -> Result<()> {
        if let Some(emb) = embedder {
            info!("MemoryGraph: using provided embedder");
            self.embedder = Some(emb);
        } else {
            let emb = crate::embedder::create_embedder()
                .map_err(|e| anyhow::anyhow!("Failed to create embedding model: {}", e))?;
            info!("MemoryGraph: embedding model loaded");
            self.embedder = Some(emb);
        }
        Ok(())
    }

    // ─── GQL Helpers ──────────────────────────────────────────────────────

    /// Escape a string value for use in a GQL literal (single-quoted).
    fn gql_escape(s: &str) -> String {
        s.replace('\\', "\\\\").replace('\'', "\\'")
    }

    /// Build a GQL property map string from a list of (key, value) pairs.
    fn gql_props(props: &[(&str, &str)]) -> String {
        let parts: Vec<String> = props
            .iter()
            .map(|(k, v)| format!("{}: '{}'", k, Self::gql_escape(v)))
            .collect();
        format!("{{{}}}", parts.join(", "))
    }

    /// Format a `serde_json::Value` as a native GQL literal.
    ///
    /// SeleneDB's GQL parser only accepts SCALAR literals in `SET`/`INSERT`
    /// property maps — list (`[...]`) and object (`{...}`) literals fail with
    /// `expected prop_ident`. Arrays/objects are therefore serialized to a
    /// JSON-encoded single-quoted string (preserving the data as a scalar
    /// text property), so graph writes with structured properties (e.g.
    /// StepDefinition `arg_template`, `depends_on`) never break the parser.
    fn format_value_as_gql(val: &serde_json::Value) -> String {
        match val {
            serde_json::Value::String(s) => format!("'{}'", Self::gql_escape(s)),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            serde_json::Value::Null => "null".to_string(),
            // Complex values: emit as a JSON-encoded scalar string literal.
            serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                let json = serde_json::to_string(val).unwrap_or_else(|_| "null".to_string());
                format!("'{}'", Self::gql_escape(&json))
            }
        }
    }

    /// Envelope variant of `parse_node_from_ref_row` — resolves a `NodeRef`
    /// binding into an open `AttrNode` via its resolved properties.
    fn attr_node_from_ref_row(
        row: &selene_db_gql::runtime::Binding,
        table: &selene_db_gql::runtime::BindingTable,
        graph_db: &GraphDb,
    ) -> Option<AttrNode> {
        Self::attr_node_from_ref_col("n", row, table, graph_db)
    }

    /// Envelope variant of `parse_node_from_ref_col` — reads the node ref from
    /// an arbitrary column (used by traversal, where endpoints land in the `m`
    /// column) and hydrates an `AttrNode` from its resolved properties.
    fn attr_node_from_ref_col(
        col: &str,
        row: &selene_db_gql::runtime::Binding,
        table: &selene_db_gql::runtime::BindingTable,
        graph_db: &GraphDb,
    ) -> Option<AttrNode> {
        use selene_db_core::value::Value;
        let n_idx = table.column_index(crate::graph::to_db_string(col))?;
        let node_ref = row.get(n_idx)?;
        let node_id = match node_ref {
            Value::NodeRef(nid) => *nid,
            _ => return None,
        };
        let props = graph_db.resolve_node_properties(node_id).ok()?;
        Self::attr_node_from_resolved(&props)
    }

    fn parse_edge_from_row(
        row: &selene_db_gql::runtime::Binding,
        table: &selene_db_gql::runtime::BindingTable,
        graph_db: &GraphDb,
    ) -> Option<GraphEdge> {
        use selene_db_core::value::Value;

        let e_idx = table.column_index(crate::graph::to_db_string("e"))?;
        let from_idx = table.column_index(crate::graph::to_db_string("from_uuid"))?;
        let to_idx = table.column_index(crate::graph::to_db_string("to_uuid"))?;

        let edge_record = row.get(e_idx)?;
        let props: Vec<(selene_db_core::db_string::DbString, Value)> = match edge_record {
            Value::Record(rec) => match rec.as_ref() {
                selene_db_core::value::Record::Open(fields) => fields.iter().cloned().collect(),
                _ => return None,
            },
            Value::EdgeRef(edge_id) => {
                let pm = graph_db.edge_properties(edge_id.clone())?;
                pm.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
            }
            _ => return None,
        };

        let from_uuid = row
            .get(from_idx)
            .and_then(|v| {
                if let Value::String(s) = v {
                    Some(s.to_string())
                } else {
                    None
                }
            })
            .unwrap_or_default();
        let to_uuid = row
            .get(to_idx)
            .and_then(|v| {
                if let Value::String(s) = v {
                    Some(s.to_string())
                } else {
                    None
                }
            })
            .unwrap_or_default();

        Self::graph_edge_from_props(&props, from_uuid, to_uuid)
    }

    /// Build a `GraphEdge` from an edge's resolved property list + endpoint
    /// UUIDs. Used by `parse_edge_from_row` and `edge_from_id` (traversal).
    fn graph_edge_from_props(
        props: &[(
            selene_db_core::db_string::DbString,
            selene_db_core::value::Value,
        )],
        from_uuid: String,
        to_uuid: String,
    ) -> Option<GraphEdge> {
        use selene_db_core::value::Value;

        let get_str = |key: &str| -> Option<String> {
            let db_key = selene_db_core::db_string::DbString::try_from(key).ok()?;
            props.iter().find(|(k, _)| k == &db_key).and_then(|(_, v)| {
                if let Value::String(s) = v {
                    let s = s.to_string();
                    if s.is_empty() {
                        None
                    } else {
                        Some(s)
                    }
                } else {
                    None
                }
            })
        };

        let uuid = get_str(PROP_UUID)?;
        let edge_type_str = get_str(PROP_EDGE_TYPE)?;
        let edge_type = serde_json::from_str::<RelationshipType>(&format!("\"{}\"", edge_type_str))
            .unwrap_or(RelationshipType::Unknown);

        let weight = {
            let db_key = selene_db_core::db_string::DbString::try_from(PROP_WEIGHT).ok()?;
            props.iter().find(|(k, _)| k == &db_key).and_then(|(_, v)| {
                if let Value::Float(f) = v {
                    Some(*f)
                } else if let Value::Int(i) = v {
                    Some(*i as f64)
                } else {
                    None
                }
            })
        };

        let created_at = get_str(PROP_CREATED_AT)
            .and_then(|s| s.parse::<DateTime<Utc>>().ok())
            .unwrap_or_else(Utc::now);

        let known_keys: [&str; 4] = [PROP_UUID, PROP_EDGE_TYPE, PROP_WEIGHT, PROP_CREATED_AT];
        let mut properties = HashMap::new();
        for (key, val) in props.iter() {
            let key_str = key.to_string();
            if !known_keys.contains(&key_str.as_str()) {
                if let Some(json_val) = selene_value_to_json(val) {
                    properties.insert(key_str, json_val);
                }
            }
        }

        Some(GraphEdge {
            id: uuid,
            edge_type,
            from_id: from_uuid,
            to_id: to_uuid,
            properties,
            created_at,
            weight,
        })
    }

    /// Resolve a GQL `EdgeRef(EdgeId)` into a `GraphEdge` (used by traversal,
    /// where the edge list contains EdgeRefs without embedded endpoint UUIDs).
    fn edge_from_id(
        edge_id: selene_db_core::identity::EdgeId,
        graph_db: &GraphDb,
    ) -> Option<GraphEdge> {
        use selene_db_core::value::Value;

        let pm = graph_db.edge_properties(edge_id.clone())?;
        let (from_node_id, to_node_id) = graph_db.edge_endpoints(edge_id.clone())?;

        // Resolve endpoint node UUIDs via their property maps.
        let uuid_of = |node_id| -> Option<String> {
            let props = graph_db.resolve_node_properties(node_id).ok()?;
            let key = selene_db_core::db_string::DbString::try_from(PROP_UUID).ok()?;
            match props.get(key.as_str()) {
                Some(Value::String(s)) if !s.to_string().is_empty() => Some(s.to_string()),
                _ => None,
            }
        };
        let from_uuid = uuid_of(from_node_id)?;
        let to_uuid = uuid_of(to_node_id)?;

        let props: Vec<(selene_db_core::db_string::DbString, Value)> =
            pm.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        Self::graph_edge_from_props(&props, from_uuid, to_uuid)
    }

    /// Query all SpireNode-labeled nodes and return them as envelopes.
    fn query_all_spire_nodes(&self) -> Vec<AttrNode> {
        let graph_db = match self.graph_db.as_ref() {
            Some(db) => db,
            None => return Vec::new(),
        };

        let gql = format!("MATCH (n:{}) RETURN n", LABEL_SPIRE_NODE);
        let table = match graph_db.execute_gql_query(&gql) {
            Ok(t) => t,
            _ => return Vec::new(),
        };

        table
            .rows()
            .iter()
            .filter_map(|row| Self::attr_node_from_ref_row(row, &table, graph_db))
            .collect()
    }

    /// Query a single SpireNode by UUID into the open `AttrNode` envelope.
    fn query_attr_node_by_uuid(&self, uuid: &str) -> Option<AttrNode> {
        let graph_db = self.graph_db.as_ref()?;
        let gql = format!(
            "MATCH (n:{}) WHERE n.uuid = '{}' RETURN n",
            LABEL_SPIRE_NODE,
            Self::gql_escape(uuid),
        );
        let table = graph_db.execute_gql_query(&gql).ok()?;
        let row = table.rows().first()?;
        let n_idx = table.column_index(crate::graph::to_db_string("n"))?;
        let selene_db_core::value::Value::NodeRef(nid) = row.get(n_idx)? else {
            return None;
        };
        let props = graph_db.resolve_node_properties(*nid).ok()?;
        Self::attr_node_from_resolved(&props)
    }

    /// Query edges for a node (both outgoing and incoming) via GQL.
    fn query_edges_for_node(&self, node_uuid: &str) -> Vec<GraphEdge> {
        let graph_db = match self.graph_db.as_ref() {
            Some(db) => db,
            None => return Vec::new(),
        };

        let mut edges = Vec::new();

        let outgoing_gql = format!(
            "MATCH (n:{})-[e]->(m) WHERE n.uuid = '{}' RETURN e, n.uuid AS from_uuid, m.uuid AS to_uuid",
            LABEL_SPIRE_NODE,
            Self::gql_escape(node_uuid)
        );
        if let Ok(table) = graph_db.execute_gql_query(&outgoing_gql) {
            for row in table.rows() {
                if let Some(edge) = Self::parse_edge_from_row(row, &table, graph_db) {
                    edges.push(edge);
                }
            }
        }

        let incoming_gql = format!(
            "MATCH (m)-[e]->(n:{}) WHERE n.uuid = '{}' RETURN e, m.uuid AS from_uuid, n.uuid AS to_uuid",
            LABEL_SPIRE_NODE,
            Self::gql_escape(node_uuid)
        );
        if let Ok(table) = graph_db.execute_gql_query(&incoming_gql) {
            for row in table.rows() {
                if let Some(edge) = Self::parse_edge_from_row(row, &table, graph_db) {
                    edges.push(edge);
                }
            }
        }

        edges
    }

    /// Create a GraphNode from raw parsed data. Dispatches to the correct variant
    /// based on the NodeType, populating typed fields from the properties HashMap.
    /// Apply partial updates to an existing envelope node. Preserves the
    /// existing id / timestamps, merges property maps, and bumps the version.
    fn apply_attr_updates(node: &AttrNode, updates: NodeUpdate) -> AttrNode {
        let now = Utc::now();
        let mut merged_props = node.properties.clone();
        if let Some(updates_props) = updates.properties {
            merged_props.extend(updates_props);
        }
        AttrNode {
            id: node.id.clone(),
            node_type: updates
                .node_type
                .clone()
                .unwrap_or_else(|| node.node_type.clone()),
            subtype: updates.subtype.unwrap_or_else(|| node.subtype.clone()),
            name: updates.name.clone().unwrap_or_else(|| node.name.clone()),
            description: updates
                .description
                .clone()
                .unwrap_or_else(|| node.description.clone()),
            properties: merged_props,
            embedding_id: updates
                .embedding_id
                .clone()
                .unwrap_or_else(|| node.embedding_id.clone()),
            created_at: node.created_at,
            updated_at: now,
            version: node.version + 1,
        }
    }

    /// Store a node from the open `AttrNode` envelope. The arbitrary
    /// `node_type` string is written verbatim (no enum mapping), scalar
    /// properties inline, complex (array/object) properties SET individually.
    /// `embedding` is stored as the node's vector.
    fn store_attr_node_via_gql(&self, attr: &AttrNode, embedding: Option<&[f32]>) -> Result<()> {
        let graph_db = self
            .graph_db
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

        let subtype_val = attr
            .subtype
            .clone()
            .or_else(|| {
                attr.properties
                    .get("subtype")
                    .and_then(|v| v.as_str().map(String::from))
            })
            .unwrap_or_default();

        let mut parts: Vec<String> = vec![
            format!("{}: '{}'", PROP_UUID, Self::gql_escape(&attr.id)),
            format!(
                "{}: '{}'",
                PROP_NODE_TYPE,
                Self::gql_escape(&attr.node_type)
            ),
            format!("{}: '{}'", PROP_NAME, Self::gql_escape(&attr.name)),
            format!(
                "{}: '{}'",
                PROP_DESCRIPTION,
                Self::gql_escape(attr.description.as_deref().unwrap_or(""))
            ),
            format!("{}: '{}'", PROP_SUBTYPE, Self::gql_escape(&subtype_val)),
            format!(
                "{}: '{}'",
                PROP_EMBEDDING_ID,
                Self::gql_escape(attr.embedding_id.as_deref().unwrap_or(""))
            ),
            format!(
                "{}: '{}'",
                PROP_CREATED_AT,
                Self::gql_escape(&attr.created_at.to_rfc3339())
            ),
            format!(
                "{}: '{}'",
                PROP_UPDATED_AT,
                Self::gql_escape(&attr.updated_at.to_rfc3339())
            ),
            format!("{}: {}", PROP_VERSION, attr.version),
        ];

        // Inline scalar properties; complex ones are SET after the INSERT.
        const BASE_KEYS: [&str; 9] = [
            PROP_UUID,
            PROP_NODE_TYPE,
            PROP_NAME,
            PROP_DESCRIPTION,
            PROP_SUBTYPE,
            PROP_EMBEDDING_ID,
            PROP_CREATED_AT,
            PROP_UPDATED_AT,
            PROP_VERSION,
        ];
        let mut complex: Vec<(&String, &serde_json::Value)> = Vec::new();
        for (key, val) in &attr.properties {
            if BASE_KEYS.contains(&key.as_str()) {
                continue;
            }
            match val {
                serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                    complex.push((key, val))
                }
                _ => parts.push(format!("{}: {}", key, Self::format_value_as_gql(val))),
            }
        }

        let props_str = format!("{{{}}}", parts.join(", "));
        let create_gql = format!("INSERT (n:{} {})", LABEL_SPIRE_NODE, props_str);
        graph_db.execute_gql_write(&create_gql)?;

        for (key, val) in complex {
            let set_gql = format!(
                "MATCH (n:{}) WHERE n.{} = '{}' SET n.{} = {}",
                LABEL_SPIRE_NODE,
                PROP_UUID,
                Self::gql_escape(&attr.id),
                key,
                Self::format_value_as_gql(val),
            );
            graph_db.execute_gql_write(&set_gql)?;
        }

        if let Some(embedding) = embedding {
            let vec_str: Vec<String> = embedding.iter().map(|v| v.to_string()).collect();
            let set_gql = format!(
                "MATCH (n) WHERE n.{} = '{}' SET n.embedding = [{}]",
                PROP_UUID,
                Self::gql_escape(&attr.id),
                vec_str.join(", "),
            );
            graph_db.execute_gql_write(&set_gql)?;
        }

        Ok(())
    }

    /// Build the open `AttrNode` envelope from a node's resolved property map,
    /// preserving the arbitrary `node_type` discriminator string.
    fn attr_node_from_resolved(props: &HashMap<String, Value>) -> Option<AttrNode> {
        let get_str = |key: &str| -> Option<String> {
            props.get(key).and_then(|v| {
                if let Value::String(s) = v {
                    let s = s.to_string();
                    if s.is_empty() {
                        None
                    } else {
                        Some(s)
                    }
                } else {
                    None
                }
            })
        };
        let get_str_or_empty = |key: &str| -> String { get_str(key).unwrap_or_default() };

        let id = get_str(PROP_UUID)?;
        let node_type = get_str(PROP_NODE_TYPE)?;
        let name = get_str_or_empty(PROP_NAME);
        let subtype = get_str(PROP_SUBTYPE).filter(|s| !s.is_empty());
        let description = get_str(PROP_DESCRIPTION);
        let embedding_id = get_str(PROP_EMBEDDING_ID);

        let created_at = get_str(PROP_CREATED_AT)
            .and_then(|s| s.parse::<DateTime<Utc>>().ok())
            .unwrap_or_else(Utc::now);
        let updated_at = get_str(PROP_UPDATED_AT)
            .and_then(|s| s.parse::<DateTime<Utc>>().ok())
            .unwrap_or_else(Utc::now);
        let version = props
            .get(PROP_VERSION)
            .and_then(|v| {
                if let Value::Int(i) = v {
                    Some(*i as u32)
                } else {
                    None
                }
            })
            .unwrap_or(1);

        // Everything except the base keys is a domain property.
        const BASE_KEYS: [&str; 9] = [
            PROP_UUID,
            PROP_NODE_TYPE,
            PROP_NAME,
            PROP_DESCRIPTION,
            PROP_SUBTYPE,
            PROP_EMBEDDING_ID,
            PROP_CREATED_AT,
            PROP_UPDATED_AT,
            PROP_VERSION,
        ];
        let mut properties = HashMap::new();
        for (key, val) in props.iter() {
            let key_str = key.to_string();
            if !BASE_KEYS.contains(&key_str.as_str()) {
                if let Some(json_val) = selene_value_to_json(val) {
                    properties.insert(key_str, json_val);
                }
            }
        }

        Some(AttrNode {
            id,
            node_type,
            subtype,
            name,
            description,
            properties,
            embedding_id,
            created_at,
            updated_at,
            version,
        })
    }

    /// Store an edge in SeleneDB via GQL.

    fn store_edge_via_gql(
        &self,
        from_uuid: &str,
        predicate: &str,
        to_uuid: &str,
        properties: &[(&str, &str)],
    ) -> Result<()> {
        let graph_db = self
            .graph_db
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

        let props_str = Self::gql_props(properties);

        let gql = format!(
            "MATCH (a), (b) WHERE a.uuid = '{}' AND b.uuid = '{}' INSERT (a)-[e:{} {}]->(b)",
            Self::gql_escape(from_uuid),
            Self::gql_escape(to_uuid),
            predicate,
            props_str,
        );
        graph_db.execute_gql_write(&gql)?;
        Ok(())
    }

    /// Delete a node and all its edges via GQL.
    fn delete_node_via_gql(&self, uuid: &str) -> Result<()> {
        let graph_db = self
            .graph_db
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

        let gql = format!(
            "MATCH (n) WHERE n.uuid = '{}' DETACH DELETE n",
            Self::gql_escape(uuid),
        );
        graph_db.execute_gql_write(&gql)?;
        Ok(())
    }

    /// Delete an edge by UUID via GQL.
    fn delete_edge_via_gql(&self, uuid: &str) -> Result<()> {
        let graph_db = self
            .graph_db
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

        let gql = format!(
            "MATCH ()-[e]->() WHERE e.uuid = '{}' DELETE e",
            Self::gql_escape(uuid),
        );
        graph_db.execute_gql_write(&gql)?;
        Ok(())
    }

    /// Query nodes by discriminator / subtype / name via GQL into the open
    /// `AttrNode` envelope.
    fn query_attr_nodes(
        &self,
        node_type: Option<&str>,
        subtype: Option<&str>,
        name: Option<&str>,
        limit: Option<u32>,
    ) -> Vec<AttrNode> {
        let graph_db = match self.graph_db.as_ref() {
            Some(db) => db,
            None => return Vec::new(),
        };

        let mut conditions: Vec<String> = Vec::new();
        if let Some(nt) = node_type {
            conditions.push(format!("n.{} = '{}'", PROP_NODE_TYPE, Self::gql_escape(nt)));
        }
        if let Some(st) = subtype {
            conditions.push(format!("n.{} = '{}'", PROP_SUBTYPE, Self::gql_escape(st)));
        }
        if let Some(nm) = name {
            conditions.push(format!("n.{} = '{}'", PROP_NAME, Self::gql_escape(nm)));
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", conditions.join(" AND "))
        };

        let limit_clause = limit.map(|l| format!(" LIMIT {}", l)).unwrap_or_default();

        let gql = format!(
            "MATCH (n:{}){} RETURN n{}",
            LABEL_SPIRE_NODE, where_clause, limit_clause,
        );

        match graph_db.execute_gql_query(&gql) {
            Ok(table) => table
                .rows()
                .iter()
                .filter_map(|row| Self::attr_node_from_ref_row(row, &table, graph_db))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Check whether adding a `depends_on` edge would create a cycle.
    fn would_create_cycle(&self, from_id: &str, to_id: &str) -> bool {
        if from_id == to_id {
            return true;
        }

        let graph_db = match self.graph_db.as_ref() {
            Some(db) => db,
            None => return false,
        };

        let gql = format!(
            "MATCH (a)-[e:DependsOn*]->(b) WHERE a.uuid = '{}' AND b.uuid = '{}' RETURN a, b LIMIT 1",
            Self::gql_escape(to_id),
            Self::gql_escape(from_id),
        );

        match graph_db.execute_gql_query(&gql) {
            Ok(table) => !table.rows().is_empty(),
            _ => false,
        }
    }

    /// BFS traversal from a start node using GQL.
    fn traverse(&self, start_node_id: &str, options: &TraversalOptions) -> Result<TraversalResult> {
        let graph_db = self
            .graph_db
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

        let arrow = match &options.direction {
            Some(TraversalDirection::Out) => "->",
            Some(TraversalDirection::In) => "<-",
            Some(TraversalDirection::Both) => "-",
            None => "-",
        };

        let max_depth = options.max_depth;
        // SeleneDB GQL 1.4 rejects `[e*1..1]` (equal lower/upper bounds) — for
        // depth 1 emit a plain single-edge pattern instead. Note the leading
        // `-` connecting `(n)` to `[e]` (the trailing arrow sets direction).
        let edge_pattern = if max_depth == 1 {
            format!("-[e]{}", arrow)
        } else if options
            .relationship_types
            .as_ref()
            .is_none_or(|v| v.is_empty())
        {
            format!("-[e*1..{}]{}", max_depth, arrow)
        } else {
            let types: Vec<String> = options
                .relationship_types
                .as_ref()
                .unwrap()
                .iter()
                .map(relationship_type_to_gql_label)
                .collect();
            format!("-[e*1..{}{}]{}", max_depth, types.join("|"), arrow)
        };

        // Variable-length match WITHOUT a `path =` binding (the path-selector
        // syntax is unsupported by SeleneDB GQL 1.4). Each row is (n, m, e):
        // n = the start node, m = a reachable endpoint, e = the List of
        // EdgeRefs along that path (flattened + deduped below).
        let gql = format!(
            "MATCH (n:{}){}(m) WHERE n.uuid = '{}' RETURN n, m, e",
            LABEL_SPIRE_NODE,
            edge_pattern,
            Self::gql_escape(start_node_id),
        );

        let table = match graph_db.execute_gql_query(&gql) {
            Ok(t) => t,
            _ => {
                return Ok(TraversalResult {
                    nodes: Vec::new(),
                    edges: Vec::new(),
                    paths: Vec::new(),
                })
            }
        };

        let mut nodes: HashMap<String, AttrNode> = HashMap::new();
        let mut edges: Vec<GraphEdge> = Vec::new();
        let mut seen_edge_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let paths: Vec<TraversalPath> = Vec::new();

        for row in table.rows() {
            // Start node (n) and each reachable endpoint (m) are NodeRefs.
            if let Some(node) = Self::attr_node_from_ref_row(row, &table, graph_db) {
                nodes.entry(node.id().to_string()).or_insert(node);
            }
            if let Some(node) = Self::attr_node_from_ref_col("m", row, &table, graph_db) {
                nodes.entry(node.id().to_string()).or_insert(node);
            }

            // The path's edges: a single `EdgeRef` for `[e]` (depth 1) or a
            // `List` of `EdgeRef`s for variable-length `[e*1..N]`.
            if let Some(e_idx) = table.column_index(crate::graph::to_db_string("e")) {
                match row.get(e_idx) {
                    Some(selene_db_core::value::Value::EdgeRef(edge_id)) => {
                        if let Some(edge) = Self::edge_from_id(edge_id.clone(), graph_db) {
                            if seen_edge_ids.insert(edge.id.clone()) {
                                edges.push(edge);
                            }
                        }
                    }
                    Some(selene_db_core::value::Value::List(edge_refs)) => {
                        for edge_ref in edge_refs {
                            if let selene_db_core::value::Value::EdgeRef(edge_id) = edge_ref {
                                if let Some(edge) = Self::edge_from_id(edge_id.clone(), graph_db) {
                                    if seen_edge_ids.insert(edge.id.clone()) {
                                        edges.push(edge);
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(TraversalResult {
            nodes: nodes.into_values().collect(),
            edges,
            paths,
        })
    }

    /// Find a node by (node_type, name) within an existing transaction.
    /// Schedule a debounced snapshot write.
    fn schedule_snapshot(&self) {
        if let Some(tx) = &self.snapshot_tx {
            let _ = tx.send(());
        }
    }

    /// Spawn the background debounced snapshot task.
    fn spawn_snapshot_task(
        data_dir: PathBuf,
        graph_db: Arc<GraphDb>,
        mut snapshot_rx: mpsc::UnboundedReceiver<()>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let debounce_duration = Duration::from_secs(2);

            loop {
                match snapshot_rx.recv().await {
                    Some(_) => {
                        loop {
                            tokio::select! {
                                Some(_) = snapshot_rx.recv() => {
                                    continue;
                                }
                                _ = tokio::time::sleep(debounce_duration) => {
                                    break;
                                }
                            }
                        }

                        let next_seq = match GraphDb::latest_snapshot_sequence(&data_dir) {
                            Ok(Some(seq)) => seq + 1,
                            Ok(None) => 1,
                            Err(e) => {
                                warn!(
                                    "MemoryGraph snapshot task: failed to get latest sequence: {}",
                                    e
                                );
                                continue;
                            }
                        };

                        match graph_db.write_snapshot(&data_dir, next_seq, true) {
                            Ok(outcome) => {
                                info!("MemoryGraph snapshot task: snapshot written (seq={}, sections={})", outcome.snapshot_seq, outcome.section_count);
                            }
                            Err(e) => {
                                warn!("MemoryGraph snapshot task: failed to write snapshot: {}", e);
                            }
                        }
                    }
                    None => {
                        info!("MemoryGraph snapshot task: shutting down");
                        break;
                    }
                }
            }
        })
    }

    /// Serialize a typed `GraphNode::Platform` into the generic JSON shape
    /// `{ "id", "name", "properties": {flat map} }` that crosses the crate
    /// boundary (spire-core rebuilds its typed `Platform` view from this).
    /// `{ "id", "name", "properties": {flat map} }` that crosses the crate
    /// boundary (spire-core rebuilds its typed `Platform` view from this).
    fn platform_node_to_json(node: &AttrNode) -> Option<serde_json::Value> {
        if node.node_type_str() != "Platform" {
            return None;
        }
        let mut props = serde_json::Map::new();
        for key in [
            "os",
            "cpu_family",
            "cpu",
            "endian",
            "target_triple",
            "march",
            "c_compiler",
            "cpp_compiler",
            "ar",
            "strip",
            "ld",
            "pkgconfig",
            "c_args_extra",
            "cpp_args_extra",
            "linker_args_extra",
            "needs_exe_wrapper",
            "sysroot_root",
            "sysroot_lib_dirs",
            "sysroot_include_dirs",
            "sysroot_pkg_config_libdir",
        ] {
            if let Some(v) = node.get(key) {
                props.insert(key.to_string(), v.clone());
            }
        }
        let name = node.name().to_string();
        Some(serde_json::json!({
            "id": node.name(),
            "name": name,
            "properties": props,
        }))
    }

    // ─── Async Handlers ─────────────────────────────────

    async fn handle_search_context(
        &self,
        query: String,
        options: Option<SearchOptions>,
    ) -> Result<ContextSearchResult> {
        let embedder = self
            .embedder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Embedder not initialized"))?;
        let graph_db = self
            .graph_db
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

        let embedding = embedder.embed(&query).await?;
        let k = options.as_ref().and_then(|o| o.top_k).unwrap_or(10);
        let hits = graph_db.exact_vector_search_nodes(
            LABEL_SPIRE_NODE,
            "embedding",
            &embedding.vector,
            VectorMetric::Cosine,
            k,
        )?;

        let mut scored_nodes = Vec::new();
        for hit in &hits {
            let gql = format!("MATCH (n) WHERE id(n) = {} RETURN n.uuid", hit.0);
            if let Ok(table) = graph_db.execute_gql_query(&gql) {
                if let Some(row) = table.rows().first() {
                    if let Some(uuid_idx) = table.column_index(crate::graph::to_db_string("n.uuid"))
                    {
                        if let Some(Value::String(uuid)) = row.get(uuid_idx) {
                            if let Some(node) = self.query_attr_node_by_uuid(uuid.as_ref()) {
                                scored_nodes.push(ScoredNode {
                                    node,
                                    similarity: hit.1,
                                    source: RetrievalSource::Semantic,
                                    score: hit.1,
                                });
                            }
                        }
                    }
                }
            }
        }

        let total = scored_nodes.len();
        Ok(ContextSearchResult {
            nodes: scored_nodes,
            relationships: Vec::new(),
            total_results: total,
            search_time_ms: 0,
            truncated: false,
        })
    }

    async fn handle_add_memory(
        &self,
        text: String,
        metadata: Option<MemoryMetadata>,
    ) -> Result<MemoryEntry> {
        let embedder = self
            .embedder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Embedder not initialized"))?;
        let now = Utc::now();
        let memory_id = Uuid::new_v4().to_string();

        let embedding = embedder.embed(&text).await?;

        let _mem_type_str = metadata.as_ref().and_then(|m| m.mem_type.as_ref());
        let attr = AttrNode {
            id: memory_id.clone(),
            node_type: "Memory".to_string(),
            subtype: None,
            name: text.chars().take(100).collect(),
            description: Some(text.clone()),
            properties: HashMap::new(),
            embedding_id: Some(memory_id.clone()),
            created_at: now,
            updated_at: now,
            version: 1,
        };

        self.store_attr_node_via_gql(&attr, Some(&embedding.vector))?;
        self.schedule_snapshot();

        let default_metadata = MemoryMetadata {
            mem_type: None,
            tags: None,
            source: None,
            confidence: None,
        };
        let mem_id = memory_id.clone();
        Ok(MemoryEntry {
            id: mem_id,
            text,
            metadata: metadata.unwrap_or(default_metadata),
            embedding_id: memory_id.clone(),
            node_id: Some(memory_id),
            created_at: now,
            updated_at: now,
        })
    }

    async fn handle_recall(&self, query: String, limit: Option<usize>) -> Result<Vec<MemoryEntry>> {
        let embedder = self
            .embedder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Embedder not initialized"))?;
        let graph_db = self
            .graph_db
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

        let embedding = embedder.embed(&query).await?;
        let k = limit.unwrap_or(10);

        let hits = graph_db.exact_vector_search_nodes(
            LABEL_SPIRE_NODE,
            "embedding",
            &embedding.vector,
            VectorMetric::Cosine,
            k,
        )?;

        let mut entries = Vec::new();
        for hit in &hits {
            let gql = format!("MATCH (n) WHERE id(n) = {} RETURN n", hit.0);
            if let Ok(table) = graph_db.execute_gql_query(&gql) {
                if let Some(row) = table.rows().first() {
                    if let Some(node) = Self::attr_node_from_ref_row(row, &table, graph_db) {
                        let default_metadata = MemoryMetadata {
                            mem_type: None,
                            tags: None,
                            source: None,
                            confidence: None,
                        };
                        let node_id = node.id().to_string();
                        entries.push(MemoryEntry {
                            id: node_id,
                            text: node
                                .description()
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| node.name().to_string()),
                            metadata: default_metadata,
                            embedding_id: node.embedding_id().unwrap_or("").to_string(),
                            node_id: Some(node.id().to_string()),
                            created_at: node.created_at(),
                            updated_at: node.updated_at(),
                        });
                    }
                }
            }
        }

        Ok(entries)
    }

    // ─── Spatial Queries ─────────────────────────────────────────────────

    /// Run a single spatial query and return the matching nodes.
    ///
    /// All variants pre-filter with a GQL bounding-box range scan over the
    /// scalar spatial columns, then refine with the exact predicates in
    /// `crate::spatial`. See [`SpatialQuery`] for the semantics of each
    /// variant and the `AttrNode` spatial helpers for how nodes carry
    /// location data.
    fn run_spatial_query(
        &self,
        query: &SpatialQuery,
        node_type: Option<&str>,
        subtype: Option<&str>,
        limit: Option<usize>,
    ) -> Result<SpatialQueryResult> {
        self.graph_db
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

        match query {
            SpatialQuery::BoundingBox { rect } => {
                self.spatial_bounding_box(rect, node_type, subtype, limit)
            }
            SpatialQuery::Radius {
                center,
                radius_meters,
            } => self.spatial_radius(center, *radius_meters, node_type, subtype, limit),
            SpatialQuery::Nearest { center, k } => {
                self.spatial_nearest(center, *k, node_type, subtype)
            }
            SpatialQuery::Contains { geometry } => {
                self.spatial_contains(geometry, node_type, subtype, limit)
            }
            SpatialQuery::Intersects { geometry } => {
                self.spatial_intersects(geometry, node_type, subtype, limit)
            }
        }
    }

    /// Extra `WHERE` clauses restricting candidates by node type / subtype.
    fn spatial_type_filter(node_type: Option<&str>, subtype: Option<&str>) -> String {
        let mut filter = String::new();
        if let Some(nt) = node_type {
            filter.push_str(&format!(
                " AND n.{} = '{}'",
                PROP_NODE_TYPE,
                Self::gql_escape(nt)
            ));
        }
        if let Some(st) = subtype {
            filter.push_str(&format!(
                " AND n.{} = '{}'",
                PROP_SUBTYPE,
                Self::gql_escape(st)
            ));
        }
        filter
    }

    /// Fetch every node whose stored spatial footprint (the bounding-box
    /// columns, or plain `latitude`/`longitude` columns) intersects `window`.
    ///
    /// This is deliberately a *superset* pre-filter; the callers apply the
    /// exact spatial predicate afterwards. Two GQL range queries are used (one
    /// per storage layout) and merged/deduped by UUID.
    fn spatial_candidates_in_window(
        &self,
        window: &geo::Rect<f64>,
        node_type: Option<&str>,
        subtype: Option<&str>,
    ) -> Vec<AttrNode> {
        let graph_db = match self.graph_db.as_ref() {
            Some(db) => db,
            None => return Vec::new(),
        };
        let (min_lng, min_lat) = (window.min().x, window.min().y);
        let (max_lng, max_lat) = (window.max().x, window.max().y);
        let filter = Self::spatial_type_filter(node_type, subtype);

        // Nodes carrying the pre-computed bounding-box columns.
        let bbox_cond = format!(
            "n.{p_min_lng} <= {max_lng} AND n.{p_max_lng} >= {min_lng} \
             AND n.{p_min_lat} <= {max_lat} AND n.{p_max_lat} >= {min_lat}{filter}",
            p_min_lng = crate::spatial::PROP_MIN_LNG,
            p_max_lng = crate::spatial::PROP_MAX_LNG,
            p_min_lat = crate::spatial::PROP_MIN_LAT,
            p_max_lat = crate::spatial::PROP_MAX_LAT,
        );
        // Nodes carrying plain point columns (legacy layout).
        let point_cond = format!(
            "n.{p_lng} >= {min_lng} AND n.{p_lng} <= {max_lng} \
             AND n.{p_lat} >= {min_lat} AND n.{p_lat} <= {max_lat}{filter}",
            p_lng = crate::spatial::PROP_LONGITUDE,
            p_lat = crate::spatial::PROP_LATITUDE,
        );

        let mut out: Vec<AttrNode> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for cond in [bbox_cond, point_cond] {
            let gql = format!("MATCH (n:{}) WHERE {} RETURN n", LABEL_SPIRE_NODE, cond);
            if let Ok(table) = graph_db.execute_gql_query(&gql) {
                for row in table.rows() {
                    if let Some(attr) = Self::attr_node_from_ref_row(row, &table, graph_db) {
                        if seen.insert(attr.id.clone()) {
                            out.push(attr);
                        }
                    }
                }
            }
        }
        out
    }

    /// The best available geometry for a node: its stored `geometry`, else the
    /// `latitude`/`longitude` point as a degenerate `Point` geometry.
    fn node_geometry(attr: &AttrNode) -> Option<geo::Geometry<f64>> {
        if let Some(g) = attr.spatial_geometry() {
            return Some(g);
        }
        attr.geo_point().map(geo::Geometry::Point)
    }

    /// Great-circle distance from `center` to a node's geometry — `0` when the
    /// geometry contains the center (e.g. a point inside a polygon zone).
    fn spatial_distance_to(center: &geo::Point<f64>, attr: &AttrNode) -> f64 {
        match Self::node_geometry(attr) {
            Some(g) => crate::spatial::distance_point_to_geometry(center, &g),
            None => f64::INFINITY,
        }
    }

    /// Optionally sort by ascending distance, cap at `limit`, and report
    /// whether the result was truncated.
    fn spatial_result(
        &self,
        mut scored: Vec<DistanceScoredNode>,
        limit: Option<usize>,
        sort_by_distance: bool,
    ) -> SpatialQueryResult {
        if sort_by_distance {
            scored.sort_by(|a, b| {
                a.distance_meters
                    .unwrap_or(f64::INFINITY)
                    .partial_cmp(&b.distance_meters.unwrap_or(f64::INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.node.id.cmp(&b.node.id))
            });
        }
        let total_results = scored.len();
        let truncated = limit.map(|l| l < scored.len()).unwrap_or(false);
        if let Some(l) = limit {
            scored.truncate(l);
        }
        SpatialQueryResult {
            nodes: scored,
            total_results,
            truncated,
        }
    }

    /// `BoundingBox`: nodes whose stored bounding box (a point's degenerate
    /// box included) lies entirely inside `rect`.
    fn spatial_bounding_box(
        &self,
        rect: &geo::Rect<f64>,
        node_type: Option<&str>,
        subtype: Option<&str>,
        limit: Option<usize>,
    ) -> Result<SpatialQueryResult> {
        let candidates = self.spatial_candidates_in_window(rect, node_type, subtype);
        let mut scored = Vec::new();
        for attr in candidates {
            let inside = match attr.geo_bounds() {
                Some(bounds) => {
                    rect.min().x <= bounds.min().x
                        && bounds.max().x <= rect.max().x
                        && rect.min().y <= bounds.min().y
                        && bounds.max().y <= rect.max().y
                }
                None => attr
                    .geo_point()
                    .map(|p| crate::spatial::point_in_rect(&p, rect))
                    .unwrap_or(false),
            };
            if inside {
                scored.push(DistanceScoredNode {
                    node: attr,
                    distance_meters: None,
                });
            }
        }
        Ok(self.spatial_result(scored, limit, false))
    }

    /// `Radius`: nodes whose geometry is within `radius_meters` of `center`
    /// (nodes whose geometry contains the center are at distance 0).
    fn spatial_radius(
        &self,
        center: &geo::Point<f64>,
        radius_meters: f64,
        node_type: Option<&str>,
        subtype: Option<&str>,
        limit: Option<usize>,
    ) -> Result<SpatialQueryResult> {
        let window = crate::spatial::bounding_box_for_radius(center, radius_meters);
        let candidates = self.spatial_candidates_in_window(&window, node_type, subtype);
        let mut scored = Vec::new();
        for attr in candidates {
            let distance = Self::spatial_distance_to(center, &attr);
            if distance <= radius_meters {
                scored.push(DistanceScoredNode {
                    node: attr,
                    distance_meters: Some(distance),
                });
            }
        }
        Ok(self.spatial_result(scored, limit, true))
    }

    /// `Nearest`: the `k` nodes closest to `center`. The candidate window
    /// doubles until it holds at least `k` candidates or spans the globe.
    fn spatial_nearest(
        &self,
        center: &geo::Point<f64>,
        k: usize,
        node_type: Option<&str>,
        subtype: Option<&str>,
    ) -> Result<SpatialQueryResult> {
        let k = k.max(1);
        let mut radius_meters = 25_000.0;
        let mut candidates: Vec<AttrNode> = Vec::new();
        while radius_meters <= 20_000_000.0 {
            let window = crate::spatial::bounding_box_for_radius(center, radius_meters);
            candidates = self.spatial_candidates_in_window(&window, node_type, subtype);
            if candidates.len() >= k {
                break;
            }
            radius_meters *= 2.0;
        }

        let scored: Vec<DistanceScoredNode> = candidates
            .into_iter()
            .map(|attr| {
                let distance = Self::spatial_distance_to(center, &attr);
                DistanceScoredNode {
                    node: attr,
                    distance_meters: Some(distance),
                }
            })
            .collect();
        Ok(self.spatial_result(scored, Some(k), true))
    }

    /// `Contains`: nodes whose geometry fully contains the query geometry.
    fn spatial_contains(
        &self,
        geometry: &geo::Geometry<f64>,
        node_type: Option<&str>,
        subtype: Option<&str>,
        limit: Option<usize>,
    ) -> Result<SpatialQueryResult> {
        let window = crate::spatial::geometry_bounds(geometry)
            .ok_or_else(|| anyhow::anyhow!("spatial query: query geometry has no bounds"))?;
        let candidates = self.spatial_candidates_in_window(&window, node_type, subtype);
        let mut scored = Vec::new();
        for attr in candidates {
            let node_geom = match Self::node_geometry(&attr) {
                Some(g) => g,
                None => continue,
            };
            if crate::spatial::geometry_contains(&node_geom, geometry) {
                scored.push(DistanceScoredNode {
                    node: attr,
                    distance_meters: None,
                });
            }
        }
        Ok(self.spatial_result(scored, limit, false))
    }

    /// `Intersects`: nodes whose geometry shares any point with the query
    /// geometry (overlap, touch, or containment either way).
    fn spatial_intersects(
        &self,
        geometry: &geo::Geometry<f64>,
        node_type: Option<&str>,
        subtype: Option<&str>,
        limit: Option<usize>,
    ) -> Result<SpatialQueryResult> {
        let window = crate::spatial::geometry_bounds(geometry)
            .ok_or_else(|| anyhow::anyhow!("spatial query: query geometry has no bounds"))?;
        let candidates = self.spatial_candidates_in_window(&window, node_type, subtype);
        let mut scored = Vec::new();
        for attr in candidates {
            let node_geom = match Self::node_geometry(&attr) {
                Some(g) => g,
                None => continue,
            };
            if crate::spatial::geometries_intersect(&node_geom, geometry) {
                scored.push(DistanceScoredNode {
                    node: attr,
                    distance_meters: None,
                });
            }
        }
        Ok(self.spatial_result(scored, limit, false))
    }

    fn store_node_in_txn(
        graph_db: &crate::graph::GraphDb,
        attr: &AttrNode,
        embedding_vector: Option<&[f32]>,
    ) -> anyhow::Result<()> {
        let node_type_stored = attr.node_type.clone();
        let description_val = attr.description().unwrap_or("");
        let subtype_val = attr.subtype().unwrap_or("");
        let embedding_id_val = attr.embedding_id().unwrap_or("");

        let created_at_str = attr.created_at().to_rfc3339();
        let updated_at_str = attr.updated_at().to_rfc3339();
        let version_str = attr.version().to_string();

        // Step 1: INSERT with standard properties only
        let parts: Vec<String> = vec![
            format!("{}: '{}'", PROP_UUID, Self::gql_escape(attr.id())),
            format!(
                "{}: '{}'",
                PROP_NODE_TYPE,
                Self::gql_escape(&node_type_stored)
            ),
            format!("{}: '{}'", PROP_NAME, Self::gql_escape(attr.name())),
            format!(
                "{}: '{}'",
                PROP_DESCRIPTION,
                Self::gql_escape(description_val)
            ),
            format!("{}: '{}'", PROP_SUBTYPE, Self::gql_escape(&subtype_val)),
            format!(
                "{}: '{}'",
                PROP_EMBEDDING_ID,
                Self::gql_escape(embedding_id_val)
            ),
            format!(
                "{}: '{}'",
                PROP_CREATED_AT,
                Self::gql_escape(&created_at_str)
            ),
            format!(
                "{}: '{}'",
                PROP_UPDATED_AT,
                Self::gql_escape(&updated_at_str)
            ),
            format!("{}: {}", PROP_VERSION, version_str),
        ];

        let props_str = format!("{{{}}}", parts.join(", "));
        let gql = format!("INSERT (n:{} {})", LABEL_SPIRE_NODE, props_str);
        graph_db.execute_gql_write(&gql)?;

        // Step 2: SET each envelope property individually (the open `AttrNode`
        // envelope carries every domain field — typed or dynamic — in one map).
        for (key, val) in &attr.properties {
            let set_gql = format!(
                "MATCH (n:{}) WHERE n.{} = '{}' SET n.{} = {}",
                LABEL_SPIRE_NODE,
                PROP_UUID,
                Self::gql_escape(attr.id()),
                key,
                Self::format_value_as_gql(val),
            );
            graph_db.execute_gql_write(&set_gql)?;
        }

        // Step 3: SET the embedding vector property if provided
        if let Some(embedding) = embedding_vector {
            let vec_str: Vec<String> = embedding.iter().map(|v| v.to_string()).collect();
            let vec_list = format!("[{}]", vec_str.join(", "));
            let set_gql = format!(
                "MATCH (n) WHERE n.{} = '{}' SET n.embedding = {}",
                PROP_UUID,
                Self::gql_escape(attr.id()),
                vec_list,
            );
            graph_db.execute_gql_write(&set_gql)?;
        }

        Ok(())
    }

    fn execute_stream_op_in_txn(
        graph_db: &crate::graph::GraphDb,
        op: &StreamOp,
    ) -> anyhow::Result<StreamOpResult> {
        match op {
            StreamOp::StoreNode(attr) => {
                Self::store_node_in_txn(graph_db, attr, None)?;
                Ok(StreamOpResult::NodeStored(attr.clone()))
            }
            StreamOp::StoreNodeWithEmbedding {
                node,
                embedding_vector,
            } => {
                Self::store_node_in_txn(graph_db, node, Some(embedding_vector))?;
                Ok(StreamOpResult::NodeStored(node.clone()))
            }
            StreamOp::UpdateNode { id, updates } => {
                let delete_gql = format!(
                    "MATCH (n) WHERE n.uuid = '{}' DETACH DELETE n",
                    Self::gql_escape(id),
                );
                graph_db.execute_gql_write(&delete_gql)?;

                let now = Utc::now();
                let updated_node = AttrNode {
                    id: id.clone(),
                    node_type: updates
                        .node_type
                        .clone()
                        .unwrap_or_else(|| "Unknown".to_string()),
                    subtype: updates.subtype.clone().unwrap_or(None),
                    name: updates.name.clone().unwrap_or_default(),
                    description: updates.description.clone().unwrap_or(None),
                    properties: updates.properties.clone().unwrap_or_default(),
                    embedding_id: updates.embedding_id.clone().unwrap_or(None),
                    created_at: now,
                    updated_at: now,
                    version: 1,
                };

                Self::store_node_in_txn(graph_db, &updated_node, None)?;
                Ok(StreamOpResult::NodeUpdated(updated_node))
            }
            StreamOp::DeleteNode(id) => {
                let gql = format!(
                    "MATCH (n) WHERE n.uuid = '{}' DETACH DELETE n",
                    Self::gql_escape(id),
                );
                graph_db.execute_gql_write(&gql)?;
                Ok(StreamOpResult::NodeDeleted)
            }
            StreamOp::CreateRelationship(rel) => {
                let edge_uuid = Uuid::new_v4().to_string();
                let now = Utc::now();
                let edge_type_stored = relationship_type_to_gql_label(&rel.edge_type);

                let created_at_str = now.to_rfc3339();
                let mut edge_props: Vec<(&str, &str)> = vec![
                    (PROP_UUID, &edge_uuid),
                    (PROP_EDGE_TYPE, &edge_type_stored),
                    (PROP_CREATED_AT, &created_at_str),
                ];

                let weight_str;
                if let Some(ref weight) = rel.weight {
                    weight_str = weight.to_string();
                    edge_props.push((PROP_WEIGHT, &weight_str));
                }

                let props_str = Self::gql_props(&edge_props);
                let gql = format!(
                    "MATCH (a), (b) WHERE a.uuid = '{}' AND b.uuid = '{}' INSERT (a)-[e:{} {}]->(b)",
                    Self::gql_escape(&rel.from_id),
                    Self::gql_escape(&rel.to_id),
                    edge_type_stored,
                    props_str,
                );
                graph_db.execute_gql_write(&gql)?;

                Ok(StreamOpResult::RelationshipCreated(GraphEdge {
                    id: edge_uuid,
                    edge_type: rel.edge_type.clone(),
                    from_id: rel.from_id.clone(),
                    to_id: rel.to_id.clone(),
                    properties: HashMap::new(),
                    created_at: now,
                    weight: rel.weight,
                }))
            }
            StreamOp::DeleteRelationship(id) => {
                let gql = format!(
                    "MATCH ()-[e]->() WHERE e.uuid = '{}' DELETE e",
                    Self::gql_escape(id),
                );
                graph_db.execute_gql_write(&gql)?;
                Ok(StreamOpResult::RelationshipDeleted)
            }
            StreamOp::SetConfig { key, value } => {
                let value_str = serde_json::to_string(value)?;
                let delete_gql = format!(
                    "MATCH (n:{}) WHERE n.{} = '{}' DELETE n",
                    LABEL_CONFIG,
                    PROP_UUID,
                    Self::gql_escape(key),
                );
                let _ = graph_db.execute_gql_write(&delete_gql);

                let props = Self::gql_props(&[(PROP_UUID, key), (PROP_CONFIG_VALUE, &value_str)]);
                let create_gql = format!("INSERT (n:{} {})", LABEL_CONFIG, props);
                graph_db.execute_gql_write(&create_gql)?;
                Ok(StreamOpResult::ConfigSet)
            }
            StreamOp::RawGql(stmt) => {
                let _output = graph_db.execute_gql_query(stmt)?;
                let json_result: Option<serde_json::Value> = None;
                Ok(StreamOpResult::RawGql(json_result))
            }
            StreamOp::MergeNode(attr) => {
                // Reuse the existing node's ID when present so relationships
                // pointing at the old node stay valid (UPSERT semantics).
                let subtype_cond = match &attr.subtype {
                    Some(st) => format!("AND n.{} = '{}'", PROP_SUBTYPE, Self::gql_escape(st)),
                    None => String::new(),
                };
                let find_gql = format!(
                    "MATCH (n:{}) WHERE n.{} = '{}' AND n.{} = '{}' {} RETURN n LIMIT 1",
                    LABEL_SPIRE_NODE,
                    PROP_NODE_TYPE,
                    Self::gql_escape(&attr.node_type),
                    PROP_NAME,
                    Self::gql_escape(&attr.name),
                    subtype_cond,
                );
                let existing_uuid = match graph_db.execute_gql_query(&find_gql) {
                    Ok(table) => table.rows().first().and_then(|row| {
                        if let Some(n_idx) = table.column_index(crate::graph::to_db_string("n")) {
                            if let Some(selene_db_core::value::Value::NodeRef(nid)) = row.get(n_idx)
                            {
                                if let Ok(props) = graph_db.resolve_node_properties(*nid) {
                                    return Self::attr_node_from_resolved(&props)
                                        .map(|a| a.id.clone());
                                }
                            }
                        }
                        None
                    }),
                    Err(_) => None,
                };
                if let Some(existing_id) = existing_uuid {
                    // Delete old edges only (keep node id stable), then re-add
                    // the node under the same UUID.
                    let delete_gql = format!(
                        "MATCH (n) WHERE n.uuid = '{}' DETACH DELETE n",
                        Self::gql_escape(&existing_id),
                    );
                    graph_db.execute_gql_write(&delete_gql)?;

                    let mut merged = attr.clone();
                    merged.id = existing_id;
                    Self::store_node_in_txn(graph_db, &merged, None)?;
                    Ok(StreamOpResult::NodeUpdated(merged))
                } else {
                    Self::store_node_in_txn(graph_db, attr, None)?;
                    Ok(StreamOpResult::NodeStored(attr.clone()))
                }
            }
            StreamOp::MergeRelationship(rel) => {
                let edge_uuid = Uuid::new_v4().to_string();
                let now = Utc::now();
                let edge_type_stored = relationship_type_to_gql_label(&rel.edge_type);

                let delete_gql = format!(
                    "MATCH (a)-[e:{}]->(b) WHERE a.uuid = '{}' AND b.uuid = '{}' DELETE e",
                    edge_type_stored,
                    Self::gql_escape(&rel.from_id),
                    Self::gql_escape(&rel.to_id),
                );
                let _ = graph_db.execute_gql_write(&delete_gql);

                let created_at_str = now.to_rfc3339();
                let mut edge_props: Vec<(&str, &str)> = vec![
                    (PROP_UUID, &edge_uuid),
                    (PROP_EDGE_TYPE, &edge_type_stored),
                    (PROP_CREATED_AT, &created_at_str),
                ];

                let weight_str;
                if let Some(ref weight) = rel.weight {
                    weight_str = weight.to_string();
                    edge_props.push((PROP_WEIGHT, &weight_str));
                }

                let props_str = Self::gql_props(&edge_props);
                let gql = format!(
                    "MATCH (a), (b) WHERE a.uuid = '{}' AND b.uuid = '{}' INSERT (a)-[e:{} {}]->(b)",
                    Self::gql_escape(&rel.from_id),
                    Self::gql_escape(&rel.to_id),
                    edge_type_stored,
                    props_str,
                );
                graph_db.execute_gql_write(&gql)?;

                Ok(StreamOpResult::RelationshipCreated(GraphEdge {
                    id: edge_uuid,
                    edge_type: rel.edge_type.clone(),
                    from_id: rel.from_id.clone(),
                    to_id: rel.to_id.clone(),
                    properties: HashMap::new(),
                    created_at: now,
                    weight: rel.weight,
                }))
            }
            StreamOp::Commit | StreamOp::Rollback => Ok(StreamOpResult::RawGql(None)),
        }
    }
}

// ============================================================================
// Actor trait implementation
// ============================================================================

#[async_trait]
impl Actor for MemoryGraphActor {
    type Message = MemoryGraphMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            MemoryGraphMessage::Initialize { data_dir, reply_to } => {
                let result = self.init_graph(&data_dir);
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::InitializeEmbedder {
                model_path,
                embedder,
                reply_to,
            } => {
                let result = self.init_embedder(model_path, embedder);
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::StoreAttrNode { node, reply_to } => {
                let result = (|| -> Result<AttrNode> {
                    self.store_attr_node_via_gql(&node, None)?;
                    self.schedule_snapshot();
                    Ok(node)
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::GetAttrNode { id, reply_to } => {
                let result = (|| -> Result<Option<AttrNode>> {
                    let graph_db = self
                        .graph_db
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;
                    let gql = format!(
                        "MATCH (n:{}) WHERE n.uuid = '{}' RETURN n",
                        LABEL_SPIRE_NODE,
                        Self::gql_escape(&id),
                    );
                    let table = graph_db.execute_gql_query(&gql)?;
                    let mut attr = None;
                    for row in table.rows() {
                        if let Some(n_idx) = table.column_index(crate::graph::to_db_string("n")) {
                            if let Some(selene_db_core::value::Value::NodeRef(nid)) = row.get(n_idx)
                            {
                                if let Ok(props) = graph_db.resolve_node_properties(*nid) {
                                    attr = Self::attr_node_from_resolved(&props);
                                    break;
                                }
                            }
                        }
                    }
                    Ok(attr)
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::QueryAttrNodes {
                node_type,
                subtype,
                name,
                limit,
                reply_to,
            } => {
                let result = (|| -> Result<Vec<AttrNode>> {
                    let graph_db = self
                        .graph_db
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;
                    let mut conditions: Vec<String> = Vec::new();
                    if let Some(nt) = &node_type {
                        conditions.push(format!(
                            "n.{} = '{}'",
                            PROP_NODE_TYPE,
                            Self::gql_escape(nt)
                        ));
                    }
                    if let Some(st) = &subtype {
                        conditions.push(format!("n.{} = '{}'", PROP_SUBTYPE, Self::gql_escape(st)));
                    }
                    if let Some(nm) = &name {
                        conditions.push(format!("n.{} = '{}'", PROP_NAME, Self::gql_escape(nm)));
                    }
                    let where_clause = if conditions.is_empty() {
                        String::new()
                    } else {
                        format!(" WHERE {}", conditions.join(" AND "))
                    };
                    let limit_clause = limit.map(|l| format!(" LIMIT {}", l)).unwrap_or_default();
                    let gql = format!(
                        "MATCH (n:{}){} RETURN n{}",
                        LABEL_SPIRE_NODE, where_clause, limit_clause,
                    );
                    let table = graph_db.execute_gql_query(&gql)?;
                    let mut out = Vec::new();
                    for row in table.rows() {
                        if let Some(n_idx) = table.column_index(crate::graph::to_db_string("n")) {
                            if let Some(selene_db_core::value::Value::NodeRef(nid)) = row.get(n_idx)
                            {
                                if let Ok(props) = graph_db.resolve_node_properties(*nid) {
                                    if let Some(attr) = Self::attr_node_from_resolved(&props) {
                                        out.push(attr);
                                    }
                                }
                            }
                        }
                    }
                    Ok(out)
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::MergeAttrNode { node, reply_to } => {
                let result = (|| -> Result<AttrNode> {
                    let graph_db = self
                        .graph_db
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;
                    // Reuse an existing UUID for the same (node_type, subtype,
                    // name) so upserts keep relationships pointing at the old node.
                    let subtype_cond = match &node.subtype {
                        Some(st) => format!("AND n.{} = '{}'", PROP_SUBTYPE, Self::gql_escape(st)),
                        None => String::new(),
                    };
                    let gql = format!(
                        "MATCH (n:{}) WHERE n.{} = '{}' AND n.{} = '{}' {} RETURN n LIMIT 1",
                        LABEL_SPIRE_NODE,
                        PROP_NODE_TYPE,
                        Self::gql_escape(&node.node_type),
                        PROP_NAME,
                        Self::gql_escape(&node.name),
                        subtype_cond,
                    );
                    let existing_uuid = match graph_db.execute_gql_query(&gql) {
                        Ok(table) => table.rows().first().and_then(|row| {
                            if let Some(n_idx) = table.column_index(crate::graph::to_db_string("n"))
                            {
                                if let Some(selene_db_core::value::Value::NodeRef(nid)) =
                                    row.get(n_idx)
                                {
                                    if let Ok(props) = graph_db.resolve_node_properties(*nid) {
                                        return Self::attr_node_from_resolved(&props)
                                            .map(|a| a.id.clone());
                                    }
                                }
                            }
                            None
                        }),
                        Err(_) => None,
                    };
                    let stored = if let Some(id) = existing_uuid {
                        self.delete_node_via_gql(&id)?;
                        let mut merged = node;
                        merged.id = id;
                        self.store_attr_node_via_gql(&merged, None)?;
                        merged
                    } else {
                        self.store_attr_node_via_gql(&node, None)?;
                        node
                    };
                    self.schedule_snapshot();
                    Ok(stored)
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::UpdateNode {
                id,
                updates,
                reply_to,
            } => {
                let result = (|| -> Result<AttrNode> {
                    let existing = self
                        .query_attr_node_by_uuid(&id)
                        .ok_or_else(|| anyhow::anyhow!("Node not found: {}", id))?;
                    let updated = Self::apply_attr_updates(&existing, updates);

                    self.delete_node_via_gql(&id)?;
                    self.store_attr_node_via_gql(&updated, None)?;
                    self.schedule_snapshot();
                    Ok(updated)
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::DeleteNode { id, reply_to } => {
                let result: Result<()> = (|| {
                    if self.query_attr_node_by_uuid(&id).is_none() {
                        return Err(anyhow::anyhow!("Node not found: {}", id));
                    }
                    self.delete_node_via_gql(&id)
                })();
                if result.is_ok() {
                    self.schedule_snapshot();
                }
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::CreateRelationship { rel, reply_to } => {
                let result = (|| -> Result<GraphEdge> {
                    self.query_attr_node_by_uuid(&rel.from_id)
                        .ok_or_else(|| anyhow::anyhow!("Source node not found: {}", rel.from_id))?;
                    self.query_attr_node_by_uuid(&rel.to_id)
                        .ok_or_else(|| anyhow::anyhow!("Target node not found: {}", rel.to_id))?;

                    if rel.edge_type == RelationshipType::DependsOn
                        && self.would_create_cycle(&rel.from_id, &rel.to_id)
                    {
                        return Err(anyhow::anyhow!(
                            "Adding this DependsOn edge would create a cycle"
                        ));
                    }

                    let edge_uuid = Uuid::new_v4().to_string();
                    let now = Utc::now();
                    let edge_type_stored = relationship_type_to_gql_label(&rel.edge_type);

                    let created_at_str = now.to_rfc3339();
                    let mut edge_props: Vec<(&str, &str)> = vec![
                        (PROP_UUID, &edge_uuid),
                        (PROP_EDGE_TYPE, &edge_type_stored),
                        (PROP_CREATED_AT, &created_at_str),
                    ];

                    let weight_str;
                    if let Some(ref weight) = rel.weight {
                        weight_str = weight.to_string();
                        edge_props.push((PROP_WEIGHT, &weight_str));
                    }

                    self.store_edge_via_gql(
                        &rel.from_id,
                        &edge_type_stored,
                        &rel.to_id,
                        &edge_props,
                    )?;
                    self.schedule_snapshot();

                    Ok(GraphEdge {
                        id: edge_uuid,
                        edge_type: rel.edge_type,
                        from_id: rel.from_id,
                        to_id: rel.to_id,
                        properties: HashMap::new(),
                        created_at: now,
                        weight: rel.weight,
                    })
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::GetRelationships { node_id, reply_to } => {
                let edges = self.query_edges_for_node(&node_id);
                let _ = reply_to.send(Ok(edges));
            }
            MemoryGraphMessage::DeleteRelationship { id, reply_to } => {
                let result = self.delete_edge_via_gql(&id);
                if result.is_ok() {
                    self.schedule_snapshot();
                }
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::Traverse {
                start_node_id,
                options,
                reply_to,
            } => {
                let result = self.traverse(&start_node_id, &options);
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::GetProjectContext { reply_to } => {
                let result = {
                    let nodes = self.query_all_spire_nodes();
                    let stats = ProjectStats {
                        total_nodes: nodes.len(),
                        total_relationships: self
                            .graph_db
                            .as_ref()
                            .map(|g| g.edge_count())
                            .unwrap_or(0),
                        last_updated: Utc::now(),
                    };
                    let project = nodes
                        .iter()
                        .find(|n| n.node_type_str() == "Project")
                        .cloned()
                        .unwrap_or_else(|| AttrNode {
                            id: "unknown".to_string(),
                            node_type: "Project".to_string(),
                            subtype: None,
                            name: "Unknown Project".to_string(),
                            description: None,
                            properties: HashMap::new(),
                            embedding_id: None,
                            created_at: Utc::now(),
                            updated_at: Utc::now(),
                            version: 1,
                        });

                    let mut active_context_nodes: Vec<AttrNode> = nodes
                        .iter()
                        .filter(|n| n.node_type_str() == "ActiveContext")
                        .cloned()
                        .collect();
                    active_context_nodes.sort_by(|a, b| b.created_at().cmp(&a.created_at()));
                    let active_context = active_context_nodes.into_iter().next();

                    let mut milestones: Vec<AttrNode> = nodes
                        .iter()
                        .filter(|n| n.node_type_str() == "Milestone")
                        .cloned()
                        .collect();
                    milestones.sort_by(|a, b| b.created_at().cmp(&a.created_at()));

                    let mut blockers: Vec<AttrNode> = nodes
                        .iter()
                        .filter(|n| n.node_type_str() == "Blocker")
                        .cloned()
                        .collect();
                    blockers.sort_by(|a, b| b.created_at().cmp(&a.created_at()));

                    let mut recent_decisions: Vec<AttrNode> = nodes
                        .iter()
                        .filter(|n| n.node_type_str() == "Decision")
                        .cloned()
                        .collect();
                    recent_decisions.sort_by(|a, b| b.created_at().cmp(&a.created_at()));

                    let mut recent_entities: Vec<AttrNode> = nodes
                        .iter()
                        .filter(|n| n.node_type_str() == "Entity")
                        .cloned()
                        .collect();
                    recent_entities.sort_by(|a, b| b.created_at().cmp(&a.created_at()));

                    let mut standards: Vec<AttrNode> = nodes
                        .iter()
                        .filter(|n| n.node_type_str() == "Standard")
                        .cloned()
                        .collect();
                    standards.sort_by(|a, b| b.created_at().cmp(&a.created_at()));

                    Ok(ProjectSnapshot {
                        project,
                        active_context,
                        milestones,
                        blockers,
                        recent_decisions,
                        recent_entities,
                        standards,
                        stats,
                    })
                };
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::SearchContext {
                query,
                options,
                reply_to,
            } => {
                let result = self.handle_search_context(query, options).await;
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::AddMemory {
                text,
                metadata,
                reply_to,
            } => {
                let result = self.handle_add_memory(text, metadata).await;
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::Recall {
                query,
                limit,
                reply_to,
            } => {
                let result = self.handle_recall(query, limit).await;
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::SpatialQuery {
                query,
                node_type,
                subtype,
                limit,
                reply_to,
            } => {
                let result =
                    self.run_spatial_query(&query, node_type.as_deref(), subtype.as_deref(), limit);
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::SetConfig {
                key,
                value,
                reply_to,
            } => {
                let result = (|| -> Result<()> {
                    let graph_db = self
                        .graph_db
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

                    let value_str = serde_json::to_string(&value)?;

                    let delete_gql = format!(
                        "MATCH (n:{}) WHERE n.{} = '{}' DELETE n",
                        LABEL_CONFIG,
                        PROP_UUID,
                        Self::gql_escape(&key),
                    );
                    let _ = graph_db.execute_gql_write(&delete_gql);

                    let props =
                        Self::gql_props(&[(PROP_UUID, &key), (PROP_CONFIG_VALUE, &value_str)]);
                    let create_gql = format!("INSERT (n:{} {})", LABEL_CONFIG, props);
                    graph_db.execute_gql_write(&create_gql)?;
                    self.schedule_snapshot();
                    Ok(())
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::GetConfig { key, reply_to } => {
                let result = (|| -> Result<Option<serde_json::Value>> {
                    let graph_db = self
                        .graph_db
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

                    let gql = format!(
                        "MATCH (n:{}) WHERE n.{} = '{}' RETURN n.{}",
                        LABEL_CONFIG,
                        PROP_UUID,
                        Self::gql_escape(&key),
                        PROP_CONFIG_VALUE,
                    );
                    match graph_db.execute_gql_query(&gql) {
                        Ok(table) => {
                            if let Some(row) = table.rows().first() {
                                if let Some(Value::String(val)) = row.get(0) {
                                    let json: serde_json::Value =
                                        serde_json::from_str(val.as_ref())?;
                                    return Ok(Some(json));
                                }
                            }
                            Ok(None)
                        }
                        _ => Ok(None),
                    }
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::BootstrapMcpConfig {
                config_path,
                reply_to,
            } => {
                let result = (|| -> Result<()> {
                    let content = std::fs::read_to_string(&config_path)?;
                    let config: McpConfigFile = serde_json::from_str(&content)?;
                    let graph_db = self
                        .graph_db
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

                    let _ = graph_db.execute_gql_write("MATCH (n:SpireMcpConfig) DELETE n");

                    for server in config.servers {
                        let entry = McpServerConfigEntry {
                            name: server.name,
                            command: server.command,
                            args: server.args,
                            env: server.env,
                            url: server.url,
                            headers: server.headers,
                            autostart: server.autostart,
                        };
                        let entry_json = serde_json::to_string(&entry)?;
                        let props = Self::gql_props(&[
                            (PROP_UUID, &entry.name),
                            (PROP_CONFIG_VALUE, &entry_json),
                        ]);
                        let gql = format!("INSERT (n:SpireMcpConfig {})", props);
                        graph_db.execute_gql_write(&gql)?;
                    }
                    self.schedule_snapshot();
                    Ok(())
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::GetMcpConfig { reply_to } => {
                let result = (|| -> Result<Vec<McpServerConfigEntry>> {
                    let graph_db = self
                        .graph_db
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

                    let gql = format!("MATCH (n:SpireMcpConfig) RETURN n.{}", PROP_CONFIG_VALUE);
                    tracing::info!("GetMcpConfig: executing GQL: {}", gql);
                    let mut entries = Vec::new();
                    match graph_db.execute_gql_query(&gql) {
                        Ok(table) => {
                            tracing::info!(
                                "GetMcpConfig: got table with {} rows",
                                table.rows().len()
                            );
                            for (row_idx, row) in table.rows().iter().enumerate() {
                                tracing::info!("GetMcpConfig: processing row {}", row_idx);
                                if let Some(Value::String(json_str)) = row.get(0) {
                                    tracing::info!(
                                        "GetMcpConfig: found config_value: {}",
                                        json_str.to_string()
                                    );
                                    if let Ok(entry) = serde_json::from_str::<McpServerConfigEntry>(
                                        json_str.as_ref(),
                                    ) {
                                        entries.push(entry);
                                    } else {
                                        tracing::warn!("GetMcpConfig: failed to parse McpServerConfigEntry from: {}", json_str.to_string());
                                    }
                                } else {
                                    tracing::warn!("GetMcpConfig: column 0 is not a String in row {} (got {:?})", row_idx, row.get(0));
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("GetMcpConfig: query error: {}", e);
                        }
                    }

                    tracing::info!("GetMcpConfig: returning {} entries", entries.len());
                    Ok(entries)
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::BootstrapPlatforms {
                platforms,
                reply_to,
            } => {
                let result = (|| -> Result<()> {
                    let graph_db = self
                        .graph_db
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;
                    let actor = &*self;

                    // Delete all existing Platform nodes so the graph mirrors
                    // the registry exactly on every startup.
                    let _ = graph_db.execute_gql_write(
                        "MATCH (n:SpireNode) WHERE n.node_type = 'platform' DETACH DELETE n",
                    );

                    for p in &platforms {
                        let id = p
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        if id.is_empty() {
                            continue;
                        }
                        let name = p
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let props: HashMap<String, serde_json::Value> = p
                            .get("properties")
                            .and_then(|v| v.as_object())
                            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                            .unwrap_or_default();
                        let now = Utc::now();
                        let attr = AttrNode {
                            // Stable id = platform id (rpi5, rock3c, …) so the
                            // node survives bootstrap re-syncs without churning
                            // UUIDs.
                            id: id.clone(),
                            node_type: "Platform".to_string(),
                            subtype: None,
                            name: id.clone(),
                            description: Some(name.clone()),
                            properties: props,
                            embedding_id: None,
                            created_at: now,
                            updated_at: now,
                            version: 1,
                        };
                        actor.store_attr_node_via_gql(&attr, None)?;
                    }
                    self.schedule_snapshot();
                    info!(
                        "BootstrapPlatforms: stored {} platform definitions",
                        self.query_attr_nodes(Some("Platform"), None, None, None)
                            .len()
                    );
                    Ok(())
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::GetPlatforms { reply_to } => {
                let result = (|| -> Result<Vec<serde_json::Value>> {
                    let nodes = self.query_attr_nodes(Some("Platform"), None, None, None);
                    Ok(nodes
                        .iter()
                        .filter_map(Self::platform_node_to_json)
                        .collect())
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::SeedIntents {
                config_path,
                reply_to,
            } => {
                let result = async {
                    let content = std::fs::read_to_string(&config_path)
                        .map_err(|e| anyhow::anyhow!("Failed to read intents config: {}", e))?;
                    let intents_config: serde_json::Value = serde_json::from_str(&content)
                        .map_err(|e| anyhow::anyhow!("Failed to parse intents config: {}", e))?;
                    let graph_db = self.graph_db.as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

                    let intents = intents_config.get("intents")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| anyhow::anyhow!("Missing or invalid 'intents' array in config"))?;

                    let mut created_nodes: Vec<(String, String, String)> = Vec::new();

                    for intent in intents {
                        let name = intent.get("name")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| anyhow::anyhow!("Intent missing 'name' field"))?;
                        let description = intent.get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");

                        let patterns = intent.get("patterns")
                            .and_then(|v| v.as_array())
                            .map(|a| a.iter()
                                .filter_map(|p| p.as_str())
                                .collect::<Vec<_>>()
                                .join(" "))
                            .unwrap_or_default();
                        let embed_text = if patterns.is_empty() {
                            format!("{}: {}", name, description)
                        } else {
                            format!("{}: {}. Patterns: {}", name, description, patterns)
                        };

                        let uuid = format!("intent-{}", name);

                        let mut prop_parts: Vec<String> = vec![
                            format!("{}: '{}'", PROP_UUID, Self::gql_escape(&uuid)),
                            format!("{}: '{}'", PROP_NODE_TYPE, "Standard"),
                            format!("{}: '{}'", PROP_SUBTYPE, "intent"),
                            format!("{}: '{}'", PROP_NAME, Self::gql_escape(name)),
                            format!("{}: '{}'", PROP_DESCRIPTION, Self::gql_escape(description)),
                        ];

                        if let Some(patterns) = intent.get("patterns").and_then(|v| v.as_array()) {
                            let pattern_strs: Vec<String> = patterns.iter()
                                .filter_map(|p| p.as_str())
                                .map(|p| format!("'{}'", Self::gql_escape(p)))
                                .collect();
                            prop_parts.push(format!("patterns: [{}]", pattern_strs.join(", ")));
                        }

                        if let Some(priority) = intent.get("priority").and_then(|v| v.as_u64()) {
                            prop_parts.push(format!("priority: {}", priority));
                        }

                        if let Some(handler) = intent.get("handler").and_then(|v| v.as_str()) {
                            prop_parts.push(format!("handler: '{}'", Self::gql_escape(handler)));
                        }

                        if let Some(action) = intent.get("action").and_then(|v| v.as_str()) {
                            prop_parts.push(format!("action: '{}'", Self::gql_escape(action)));
                        }

                        if let Some(requires_approval) = intent.get("requires_approval").and_then(|v| v.as_bool()) {
                            prop_parts.push(format!("requires_approval: {}", requires_approval));
                        }

                        if let Some(state_reqs) = intent.get("state_requirements").and_then(|v| v.as_array()) {
                            let req_strs: Vec<String> = state_reqs.iter()
                                .filter_map(|r| r.as_str())
                                .map(|r| format!("'{}'", Self::gql_escape(r)))
                                .collect();
                            prop_parts.push(format!("state_requirements: [{}]", req_strs.join(", ")));
                        }

                        let props_str = format!("{{{}}}", prop_parts.join(", "));
                        let gql = format!("INSERT (n:SpireNode {})", props_str);
                        graph_db.execute_gql_write(&gql)?;

                        created_nodes.push((uuid, name.to_string(), embed_text));
                    }

                    let embedder_opt = self.embedder.as_ref().cloned();
                    if let Some(embedder) = embedder_opt {
                        for (uuid, name, embed_text) in &created_nodes {
                            match embedder.embed(embed_text).await {
                                Ok(embedding) => {
                                    let vec_str: Vec<String> = embedding.vector.iter()
                                        .map(|v| v.to_string())
                                        .collect();
                                    let vec_list = format!("[{}]", vec_str.join(", "));
                                    let set_gql = format!(
                                        "MATCH (n) WHERE n.{} = '{}' SET n.embedding = {}",
                                        PROP_UUID,
                                        Self::gql_escape(uuid),
                                        vec_list,
                                    );
                                    if let Err(e) = graph_db.execute_gql_write(&set_gql) {
                                        warn!("SeedIntents: failed to set embedding for intent '{}': {}", name, e);
                                    }
                                }
                                Err(e) => {
                                    warn!("SeedIntents: failed to embed intent '{}': {}", name, e);
                                }
                            }
                        }
                    } else {
                        info!("SeedIntents: no embedder available, skipping intent embeddings");
                    }

                    self.schedule_snapshot();
                    Ok::<_, anyhow::Error>(())
                }.await;
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::AtomicUpdateNode {
                id,
                updates,
                reply_to,
            } => {
                let result = (|| -> Result<AttrNode> {
                    let graph_db = self
                        .graph_db
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

                    let existing = self
                        .query_attr_node_by_uuid(&id)
                        .ok_or_else(|| anyhow::anyhow!("Node not found: {}", id))?;
                    let updated = Self::apply_attr_updates(&existing, updates);

                    graph_db.execute_gql_write(&format!(
                        "MATCH (n) WHERE n.uuid = '{}' DETACH DELETE n",
                        Self::gql_escape(&id),
                    ))?;
                    self.store_attr_node_via_gql(&updated, None)?;

                    self.schedule_snapshot();
                    Ok(updated)
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::BatchGql {
                statements,
                reply_to,
            } => {
                let result = (|| -> Result<()> {
                    let graph_db = self
                        .graph_db
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

                    for stmt in &statements {
                        graph_db.execute_gql_query(stmt)?;
                    }
                    self.schedule_snapshot();
                    Ok(())
                })();
                let _ = reply_to.send(result);
            }
            MemoryGraphMessage::OpenTransactionStream { reply_to } => {
                let graph_db = match self.graph_db.as_ref() {
                    Some(db) => db.clone(),
                    None => {
                        let _ = reply_to.send(mpsc::channel(64).0);
                        return;
                    }
                };

                let (tx, mut rx): (
                    mpsc::Sender<TransactionRequest>,
                    mpsc::Receiver<TransactionRequest>,
                ) = mpsc::channel(64);
                let snapshot_tx = self.snapshot_tx.clone();

                std::thread::spawn(move || {
                    loop {
                        let request = match rx.blocking_recv() {
                            Some(req) => req,
                            None => break,
                        };

                        match request.operation {
                            StreamOp::Commit | StreamOp::Rollback => {
                                let _ = request.reply_to.send(Ok(StreamOpResult::RawGql(None)));
                                break;
                            }
                            _ => {
                                let op_result =
                                    Self::execute_stream_op_in_txn(&graph_db, &request.operation);
                                let is_err = op_result.is_err();
                                let _ = request.reply_to.send(op_result.map_err(|e| e.to_string()));
                                if is_err {
                                    break;
                                }
                            }
                        }
                    }
                    if let Some(ref stx) = snapshot_tx {
                        let _ = stx.send(());
                    }
                });

                let _ = reply_to.send(tx);
            }
            MemoryGraphMessage::Sync { reply_to } => {
                let result = (|| -> Result<()> {
                    let data_dir = self
                        .data_dir
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("data_dir not set"))?;
                    let graph_db = self
                        .graph_db
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GraphDb not initialized"))?;

                    let next_seq = match GraphDb::latest_snapshot_sequence(data_dir)? {
                        Some(seq) => seq + 1,
                        None => 1,
                    };

                    let outcome = graph_db.write_snapshot(data_dir, next_seq, true)?;
                    info!(
                        "MemoryGraph: snapshot written (seq={}, sections={})",
                        outcome.snapshot_seq, outcome.section_count
                    );
                    graph_db.compact()?;
                    Ok(())
                })();
                let _ = reply_to.send(result);
            }
        }
    }
}

// ============================================================================
// Value Conversion Helpers
// ============================================================================

/// Convert a `selene_db_core::value::Value` to a `serde_json::Value`.
fn selene_value_to_json(val: &Value) -> Option<serde_json::Value> {
    match val {
        Value::Null => Some(serde_json::Value::Null),
        Value::Bool(b) => Some(serde_json::Value::Bool(*b)),
        Value::Int(i) => Some(serde_json::Value::Number(serde_json::Number::from(*i))),
        Value::Float(f) => serde_json::Number::from_f64(*f).map(serde_json::Value::Number),
        Value::Decimal(d) => d
            .to_string()
            .parse::<serde_json::Number>()
            .ok()
            .map(serde_json::Value::Number),
        Value::String(s) => {
            let text = s.to_string();
            // Complex values (arrays/objects) are persisted as JSON-encoded
            // scalar strings (see `format_value_as_gql`). Round-trip them back
            // to their structured form when the stored text parses as a JSON
            // array or object; otherwise it's a genuine plain string.
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                if parsed.is_array() || parsed.is_object() {
                    return Some(parsed);
                }
            }
            Some(serde_json::Value::String(text))
        }
        Value::List(list) => {
            let items: Vec<serde_json::Value> =
                list.iter().filter_map(selene_value_to_json).collect();
            Some(serde_json::Value::Array(items))
        }
        Value::Record(record) => {
            let mut obj = serde_json::Map::new();
            if let selene_db_core::value::Record::Open(fields) = record.as_ref() {
                for (k, v) in fields.iter() {
                    if let Some(json_val) = selene_value_to_json(v) {
                        obj.insert(k.to_string(), json_val);
                    }
                }
            }
            Some(serde_json::Value::Object(obj))
        }
        _ => None,
    }
}

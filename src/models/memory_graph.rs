use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use tokio::sync::oneshot;

// ============================================================================
// Transaction Stream Types
// ============================================================================

/// A single operation within an atomic transaction stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamOp {
    /// Store a new node (open `AttrNode` envelope).
    StoreNode(AttrNode),
    /// Store a new node with an optional pre-computed embedding vector.
    StoreNodeWithEmbedding {
        node: AttrNode,
        embedding_vector: Vec<f32>,
    },
    /// Update an existing node.
    UpdateNode { id: String, updates: NodeUpdate },
    /// Delete a node by UUID.
    DeleteNode(String),
    /// Create a relationship between two nodes.
    CreateRelationship(RelationshipInput),
    /// Delete a relationship by UUID.
    DeleteRelationship(String),
    /// Set a config key-value pair.
    SetConfig {
        key: String,
        value: serde_json::Value,
    },
    /// Execute a raw GQL statement.
    RawGql(String),
    /// Merge (upsert) a node by (node_type, name) uniqueness constraint.
    MergeNode(AttrNode),
    /// Merge (upsert) a relationship by (edge_type, from_id, to_id) uniqueness constraint.
    MergeRelationship(RelationshipInput),
    /// Commit the transaction and close the stream.
    Commit,
    /// Roll back the transaction and close the stream.
    Rollback,
}

/// The result of a single stream operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamOpResult {
    NodeStored(AttrNode),
    NodeUpdated(AttrNode),
    NodeDeleted,
    RelationshipCreated(GraphEdge),
    RelationshipDeleted,
    ConfigSet,
    RawGql(Option<serde_json::Value>),
}

/// A request sent through a transaction stream.
#[derive(Debug)]
pub struct TransactionRequest {
    pub operation: StreamOp,
    pub reply_to: oneshot::Sender<Result<StreamOpResult, String>>,
}

// ============================================================================
// Schema Errors
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Error)]
pub enum SchemaError {
    #[error("Duplicate node: ({type_name}, {name}) already exists")]
    DuplicateNode { type_name: String, name: String },
    #[error("Node not found: {id}")]
    NodeNotFound { id: String },
    #[error(
        "Acyclic dependency violation: adding depends_on from {from} to {to} would create a cycle"
    )]
    AcyclicDependencyViolation { from: String, to: String },
}

// ============================================================================
// MCP Server Config
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfigEntry {
    pub name: String,
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    pub env: Option<HashMap<String, String>>,
    pub url: Option<String>,
    pub headers: Option<HashMap<String, String>>,
    #[serde(default = "default_autostart")]
    pub autostart: bool,
}

fn default_autostart() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfigFile {
    pub servers: Vec<McpServerConfigEntry>,
}

// ============================================================================
// Node Types — typed enum variants replace the old `properties: HashMap`
// ============================================================================

/// The type of a graph node, used to select a labelled GQL node type and
/// as a discriminator for the AttrNode envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttrNode {
    /// Stable UUID (the symbolId / graph node id).
    pub id: String,
    /// Discriminator string (serde name of the former `NodeType`, or an
    /// arbitrary domain id for new node kinds).
    pub node_type: String,
    pub subtype: Option<String>,
    pub name: String,
    pub description: Option<String>,
    /// All domain fields. For `Unknown`-stored nodes this is the dynamic
    /// property map; for typed nodes it carries their flattened fields.
    pub properties: HashMap<String, serde_json::Value>,
    pub embedding_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub version: u32,
}

impl AttrNode {
    /// The node type discriminator as a string.
    pub fn node_type_str(&self) -> &str {
        &self.node_type
    }

    /// Stable node id.
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn subtype(&self) -> Option<&str> {
        self.subtype.as_deref()
    }

    pub fn embedding_id(&self) -> Option<&str> {
        self.embedding_id.as_deref()
    }

    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    pub fn updated_at(&self) -> DateTime<Utc> {
        self.updated_at
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    /// Read a domain property from the envelope.
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.properties.get(key)
    }

    /// Typed accessor for a flattened numeric field.
    pub fn u32_prop(&self, key: &str) -> Option<u32> {
        self.properties.get(key).and_then(|v| v.as_u64().map(|n| n as u32))
    }

    /// Typed accessor for a flattened float field.
    pub fn f64_prop(&self, key: &str) -> Option<f64> {
        self.properties.get(key).and_then(|v| v.as_f64())
    }

    /// Typed accessor for a flattened boolean field.
    pub fn bool_prop(&self, key: &str) -> Option<bool> {
        self.properties.get(key).and_then(|v| v.as_bool())
    }

    /// Typed accessor for a flattened string field.
    pub fn str_prop(&self, key: &str) -> Option<String> {
        self.properties
            .get(key)
            .and_then(|v| v.as_str().map(String::from))
    }

    /// Typed accessor for a flattened string-array field.
    pub fn str_array_prop(&self, key: &str) -> Vec<String> {
        self.properties
            .get(key)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// True when the envelope's discriminator matches (e.g. `is("astFunction")`).
    pub fn is(&self, node_type: &str) -> bool {
        self.node_type == node_type
    }

    /// Project a diagnostic node's flattened fields onto the diagnostic view.
    pub fn diagnostic(&self) -> Option<DiagnosticView> {
        if !self.is("diagnostic") {
            return None;
        }
        Some(DiagnosticView {
            message: self.str_prop("message")?,
            file: self.str_prop("file"),
            line: self.u32_prop("line"),
            column: self.u32_prop("column"),
            severity: self
                .str_prop("severity")
                .unwrap_or_else(|| "error".to_string()),
            build_type: self.str_prop("build_type").unwrap_or_default(),
            build_run_id: self.str_prop("build_run_id").unwrap_or_default(),
        })
    }
}

/// Diagnostic projection from the open envelope (crate-extraction-plan §4):
/// `{file, line, column, message, severity, buildType, buildRunId}`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiagnosticView {
    pub message: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub severity: String,
    pub build_type: String,
    pub build_run_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeUpdate {
    pub node_type: Option<String>,
    pub subtype: Option<Option<String>>,
    pub name: Option<String>,
    pub description: Option<Option<String>>,
    /// Only used for `Unknown`-type nodes.
    pub properties: Option<HashMap<String, serde_json::Value>>,
    pub embedding_id: Option<Option<String>>,
}

/// Filter for querying nodes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationshipType {
    #[serde(rename = "active_context")]
    ActiveContext,
    #[serde(rename = "has_decision")]
    HasDecision,
    #[serde(rename = "has_blocker")]
    HasBlocker,
    #[serde(rename = "has_milestone")]
    HasMilestone,
    #[serde(rename = "follows_standard")]
    FollowsStandard,
    #[serde(rename = "belongs_to")]
    BelongsTo,
    #[serde(rename = "depends_on")]
    DependsOn,
    #[serde(rename = "called_by")]
    CalledBy,
    Resolves,
    Supersedes,
    #[serde(rename = "semantically_related")]
    SemanticallyRelated,
    #[serde(rename = "conversation_context")]
    ConversationContext,
    #[serde(rename = "learned_from")]
    LearnedFrom,
    #[serde(rename = "session_worked_on")]
    SessionWorkedOn,
    #[serde(rename = "informed_by")]
    InformedBy,
    #[serde(rename = "has_diagnostic")]
    HasDiagnostic,
    #[serde(rename = "ast_child")]
    AstChild,
    #[serde(rename = "ast_calls")]
    AstCalls,
    #[serde(rename = "ast_imports")]
    AstImports,
    #[serde(rename = "ast_references")]
    AstReferences,
    Custom(String),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: String,
    pub edge_type: RelationshipType,
    pub from_id: String,
    pub to_id: String,
    pub properties: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub weight: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipInput {
    pub edge_type: RelationshipType,
    pub from_id: String,
    pub to_id: String,
    pub properties: Option<HashMap<String, serde_json::Value>>,
    pub weight: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticInfo {
    pub message: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub severity: String,
    pub build_type: String,
    pub build_run_id: String,
}

// ============================================================================
// Traversal Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalOptions {
    pub max_depth: u8,
    pub relationship_types: Option<Vec<RelationshipType>>,
    pub max_nodes: Option<usize>,
    pub direction: Option<TraversalDirection>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TraversalDirection {
    #[serde(rename = "out")]
    Out,
    #[serde(rename = "in")]
    In,
    #[serde(rename = "both")]
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalResult {
    pub nodes: Vec<AttrNode>,
    pub edges: Vec<GraphEdge>,
    pub paths: Vec<TraversalPath>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalPath {
    pub nodes: Vec<AttrNode>,
    pub edges: Vec<GraphEdge>,
}

// ============================================================================
// Context & Memory Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectSnapshot {
    pub project: AttrNode,
    pub active_context: Option<AttrNode>,
    pub milestones: Vec<AttrNode>,
    pub blockers: Vec<AttrNode>,
    pub recent_decisions: Vec<AttrNode>,
    pub recent_entities: Vec<AttrNode>,
    pub standards: Vec<AttrNode>,
    pub stats: ProjectStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStats {
    pub total_nodes: usize,
    pub total_relationships: usize,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchOptions {
    pub top_k: Option<usize>,
    pub threshold: Option<f64>,
    pub node_types: Option<Vec<String>>,
    pub max_depth: Option<u8>,
    pub include_structural: Option<bool>,
    pub recency_weight: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSearchResult {
    pub nodes: Vec<ScoredNode>,
    pub relationships: Vec<GraphEdge>,
    pub total_results: usize,
    pub search_time_ms: u64,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredNode {
    pub node: AttrNode,
    pub similarity: f64,
    pub source: RetrievalSource,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RetrievalSource {
    Semantic,
    Structural,
    Ambient,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub mem_type: Option<String>,
    pub tags: Option<Vec<String>>,
    pub source: Option<String>,
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub text: String,
    pub embedding_id: String,
    pub metadata: MemoryMetadata,
    pub node_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ============================================================================
// Build-Fix Loop Types
// ============================================================================

pub const LABEL_INTENT: &str = "SpireIntent";
pub const LABEL_ERROR_TYPE: &str = "SpireError";
pub const LABEL_FIX_STRATEGY: &str = "SpireFixStrategy";
pub const LABEL_TOOL: &str = "SpireTool";
pub const LABEL_STATE: &str = "SpireState";
pub const LABEL_PATTERN: &str = "SpirePattern";
pub const LABEL_CAPABILITY: &str = "SpireCapability";

pub const REL_MAPS_TO: &str = "MAPS_TO";
pub const REL_USES: &str = "USES";
pub const REL_REQUIRES: &str = "REQUIRES";
pub const REL_FIXED_BY: &str = "FIXED_BY";
pub const REL_HAS_PATTERN: &str = "HAS_PATTERN";
pub const REL_IS_SUBTYPE_OF: &str = "IS_SUBTYPE_OF";
pub const REL_TRIGGERS: &str = "TRIGGERS";
pub const REL_USES_TOOL: &str = "USES_TOOL";
pub const REL_VALIDATES_WITH: &str = "VALIDATES_WITH";
pub const REL_DEPENDS_ON: &str = "DEPENDS_ON";
pub const REL_PRECEDES: &str = "PRECEDES";
pub const REL_RESOLVES: &str = "RESOLVES";
pub const REL_TRANSITIONS_TO: &str = "TRANSITIONS_TO";
pub const REL_CAN_ROLLBACK_TO: &str = "CAN_ROLLBACK_TO";
pub const REL_PROVIDES: &str = "PROVIDES";
pub const REL_REQUIRES_TOOL: &str = "REQUIRES_TOOL";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    pub id: String,
    pub name: String,
    pub description: String,
    pub priority: u32,
    pub requires_approval: bool,
    pub state_requirements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorType {
    pub id: String,
    pub name: String,
    pub description: String,
    pub severity: String,
    pub detection_patterns: Vec<String>,
    pub fix_strategies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixStrategy {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub confidence_threshold: f64,
    pub success_rate: f64,
    pub execution_steps: Vec<String>,
    pub has_rollback: bool,
}

pub struct Tool {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: ToolCategory,
    pub capabilities: Vec<String>,
    pub approval_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolCategory {
    Analysis,
    Fix,
    Monitoring,
    Verification,
    Recovery,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildState {
    pub id: String,
    pub name: String,
    pub description: String,
    pub conditions: Vec<String>,
    pub rollback_state: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildContext {
    pub project_root: String,
    pub build_system: String,
    pub target: Option<String>,
    pub environment: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBuildResult {
    pub build_type: String,
    pub path: String,
    pub project_name: String,
    pub success: bool,
    pub errors: Vec<BuildError>,
    pub warnings: Vec<BuildError>,
    pub exit_code: Option<i32>,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildResult {
    pub success: bool,
    pub system_results: Vec<SystemBuildResult>,
    pub build_run_id: String,
    pub duration_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildError {
    pub error_text: String,
    pub error_type: Option<String>,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub exit_code: Option<i32>,
    pub build_type: Option<String>,
    pub diagnostic_node_id: Option<String>,
    pub file_node_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredFix {
    pub strategy: FixStrategy,
    pub confidence: f64,
    pub required_tools: Vec<String>,
    pub validation_tools: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedError {
    pub error: BuildError,
    pub build_type: String,
    pub build_path: String,
    pub file_node_id: Option<String>,
    pub diagnostic_node_id: Option<String>,
    pub fix_options: Vec<ScoredFix>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixPlan {
    pub errors: Vec<AnnotatedError>,
    pub ordered_fixes: Vec<ScoredFix>,
    pub max_iterations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildStartResult {
    pub build_id: String,
    pub success: bool,
    pub build_result: BuildResult,
    pub fix_plan: Option<FixPlan>,
    pub iteration_count: u32,
    pub max_iterations: u32,
}

// ============================================================================
// Plan Mode Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PlanStatus {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "approved")]
    Approved,
    #[serde(rename = "executing")]
    Executing,
    #[serde(rename = "paused")]
    Paused,
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "rejected")]
    Rejected,
    #[serde(rename = "failed")]
    Failed,
    #[serde(rename = "skipped")]
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStepData {
    pub description: String,
    pub step_name: String,
    pub arg_template: serde_json::Value,
    pub depends_on: Vec<u32>,
    pub uses_error_context: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStatusResult {
    pub plan_id: String,
    pub goal: String,
    pub status: PlanStatus,
    pub intent_name: Option<String>,
    pub steps: Vec<PlanStepEntry>,
    pub total_steps: u32,
    pub completed_steps: u32,
    pub failed_steps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStepEntry {
    pub id: String,
    pub order: u32,
    pub description: String,
    pub step_name: String,
    pub status: PlanStatus,
    pub result: Option<String>,
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attr_node_keeps_dynamic_properties() {
        let now = Utc::now();
        let attr = AttrNode {
            id: "n1".to_string(),
            node_type: "Unknown".to_string(),
            subtype: Some("custom".to_string()),
            name: "Target".to_string(),
            description: None,
            properties: HashMap::from([
                ("confidence".to_string(), serde_json::json!(0.7)),
                ("subtype".to_string(), serde_json::json!("custom")),
            ]),
            embedding_id: None,
            created_at: now,
            updated_at: now,
            version: 3,
        };

        assert_eq!(attr.node_type, "Unknown");
        assert_eq!(attr.subtype(), Some("custom"));
        assert_eq!(attr.get("confidence"), Some(&serde_json::json!(0.7)));
        assert_eq!(attr.id(), "n1");
        assert_eq!(attr.version(), 3);
    }

    #[test]
    fn attr_node_carries_arbitrary_node_types() {
        let now = Utc::now();
        let attr = AttrNode {
            id: "scad-1".to_string(),
            node_type: "csg.scad_node".to_string(),
            subtype: None,
            name: "module".to_string(),
            description: None,
            properties: HashMap::from([
                ("kind".to_string(), serde_json::json!("module")),
                ("depth".to_string(), serde_json::json!(2)),
            ]),
            embedding_id: None,
            created_at: now,
            updated_at: now,
            version: 1,
        };

        // A brand-new domain works through the envelope: the arbitrary
        // discriminator is preserved verbatim (no closed-enum mapping).
        assert_eq!(attr.node_type_str(), "csg.scad_node");
        assert_eq!(attr.get("kind"), Some(&serde_json::json!("module")));
    }

    #[test]
    fn attr_properties_carry_typed_domain_fields() {
        let now = Utc::now();
        let diag = AttrNode {
            id: "d1".to_string(),
            node_type: "Diagnostic".to_string(),
            subtype: Some("error".to_string()),
            name: "err".to_string(),
            description: None,
            properties: HashMap::from([
                ("message".to_string(), serde_json::json!("mismatched types")),
                ("file".to_string(), serde_json::json!("src/main.rs")),
                ("line".to_string(), serde_json::json!(42)),
                ("column".to_string(), serde_json::json!(5)),
                ("severity".to_string(), serde_json::json!("error")),
                ("build_type".to_string(), serde_json::json!("Cargo")),
                ("build_run_id".to_string(), serde_json::json!("run-1")),
            ]),
            embedding_id: None,
            created_at: now,
            updated_at: now,
            version: 1,
        };
        assert_eq!(
            diag.get("message"),
            Some(&serde_json::json!("mismatched types"))
        );
        assert_eq!(diag.get("file"), Some(&serde_json::json!("src/main.rs")));
        assert_eq!(diag.get("line"), Some(&serde_json::json!(42)));
        assert_eq!(diag.get("column"), Some(&serde_json::json!(5)));
        assert_eq!(diag.get("build_run_id"), Some(&serde_json::json!("run-1")));
        assert_eq!(diag.node_type_str(), "Diagnostic");
        assert_eq!(diag.get("severity"), Some(&serde_json::json!("error")));
    }
}

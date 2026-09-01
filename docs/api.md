# API Reference

<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (c) 2026 NatureSense -->

Complete public API of `spire-core`, organised by module. Each entry lists the
item's **signature**, its **purpose**, and any usage **notes**. Types marked
*re-export* are defined elsewhere and re-exported at the given path.

- [`actors`](#actors) — `messages`, `progress`, `tools`, `web_search`, `rag`,
  `rag_ingest`, `system_prompt`, `tool_providers`, `prompt_handler`
- [`subsystems`](#subsystems) — `chat`, `graph`, `llm`, `mcp`, `tools`
- [`modules`](#modules) — filesystem, git, process, search, terminal
- [`models`](#models) — `embedding`, `memory_graph`, `analysis`
- [`embedder`](#embedder)
- [`mcp`](#mcp) — `client`
- [`config`](#config)
- [`analyzer`](#analyzer) — `scanner`, `tree_builder`, `models`
- [`graph`](#graph)
- [`platform`](#platform)
- [`transport`](#transport) — `socket`
- [`build_types`](#build-types)

---

## actors

### `actors::messages`

Platform-owned tool metadata (moved here from `spire-actor` to keep the runtime
domain-agnostic). Re-exported at `spire_core::actors::*`.

```rust
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}
```
**Purpose:** describes a tool for LLM tool-calling (OpenAI-compatible `tools`
array) and for the tool registry.
**Notes:** `serde::Serialize`/`Deserialize` are derived; used by `LlmActor`,
`ToolsActor`, `SystemPromptActor`, all modules, and `spire-code` tool providers.

```rust
pub struct ToolMessage {
    pub tool: String,
    pub args: serde_json::Value,
    pub response_tx: Responder<serde_json::Value>,
}
```
**Purpose:** generic tool-invocation message (request/reply).
**Notes:** `Responder<T>` is `oneshot::Sender<Result<T, ActorError>>` from
`spire-actor`. Kept as part of the re-exported API.

### `actors::progress` — `ProgressActor`

```rust
pub struct ProgressActor;                    // Default constructible
pub enum ProgressMessage {
    Publish { update: ProgressUpdate },                          // broadcast
    Subscribe { reply_to: oneshot::Sender<broadcast::Receiver<ProgressUpdate>> },
}
pub enum ProgressStatus { Running, Completed, Failed }
pub struct ProgressUpdate {
    pub task_id: String, pub message: String, pub percent: f64,
    pub status: ProgressStatus, pub metadata: Option<serde_json::Value>,
}
```
**Purpose:** publish/subscribe progress for long-running operations (startup
phases, ingestion, builds). UI subscribers receive a `tokio::sync::broadcast`
stream.

### `actors::tools` — `ToolsActor`

```rust
pub struct RegisteredTool { pub info: ToolInfo, pub server: String }
pub enum ToolsMessage {
    RegisterTool      { server: String, info: ToolInfo, reply_to: Responder<()> },
    UnregisterServer  { server: String, reply_to: Responder<()> },
    ListTools         { reply_to: oneshot::Sender<Vec<ToolInfo>> },
    CallTool          { tool: String, args: Value, reply_to: Responder<Value> },
    RegisterVscodeTools { reply_to: Responder<()> },
}
pub fn ToolsActor::new(tool_router_tx: mpsc::Sender<ToolRouterMessage>) -> Self
```
**Purpose:** registration and dispatch of static tools; pre-registers the VS
Code extension tools at startup. Delegates actual calls to `ToolRouterActor`.

### `actors::web_search` (tool library — not an actor)

```rust
pub fn tool_definitions() -> Vec<ToolInfo>                 // wikipedia_search,
                                                           // wikipedia_extract, web_search
pub async fn call(tool_name: &str, args: Value) -> Result<Value, String>
pub async fn wikipedia_search(query: &str, limit: usize) -> Result<Value, String>
pub async fn wikipedia_extract(title: &str) -> Result<Value, String>
```
**Purpose:** stateless web-search tools (Wikipedia + Tavily-backed), registered by
`spire-code`'s tool builder as static tool handlers.
**Notes:** HTML extraction uses `regex` (no backreferences); Tavily API key is
read from config via `config::get_global_llm_config_key("tavily.api_key")`.

### `actors::rag` — `RagActor`

Per-domain retrieval-augmented generation. Retrieval re-embeds the query and
scores chunks with **cosine similarity**; when the embedder is the no-op
placeholder (zero vectors) it falls back to lexical Jaccard overlap.

```rust
pub enum RagMessage {
    Query { domain: String, query: String, top_k: usize,
            reply_to: oneshot::Sender<Result<Vec<RagChunkResult>>> },
    ListDomains { reply_to: oneshot::Sender<Result<Vec<RagDomainInfo>>> },
    ListManifests { project_root: PathBuf,
                    reply_to: oneshot::Sender<Result<Vec<RagManifestInfo>>> },
    IngestGraphConfig { manifest_path: PathBuf, project_root: Option<PathBuf>,
                        reply_to: oneshot::Sender<Result<IngestReport>> },
    ListSources { domain: String,
                  reply_to: oneshot::Sender<Result<Vec<rag_ingest::SourceStatus>>> },
    FindInterfaces { domain: String, query: String, top_k: usize,
                     reply_to: oneshot::Sender<Result<Vec<RagChunkResult>>> },
}

pub struct RagChunkResult { pub domain: String, pub source_path: String,
                            pub chunk_index: u32, pub text: String, pub score: f32 }
pub struct RagDomainInfo  { pub id: String, pub name: String, pub description: String,
                            pub chunk_count: u64, pub source_count: u64,
                            pub corpus_version: String, pub token_count: u64,
                            pub entity_count: u64, pub relationship_count: u64 }
pub struct RagManifestInfo { pub platform_id: String, pub domain: String,
                            pub path: String, pub corpus_version: String,
                            pub description: String }
pub struct EmbedderService(pub std::sync::Arc<dyn Embedder>); // Sized wrapper
```

Constructors: `RagActor::new(knowledge_tx, memory_graph_tx, Arc<dyn Embedder>)`,
`RagActor::new_shared(memory_graph_tx, Arc<dyn Embedder>)`, and
`RagActor::from_registry(knowledge_tx, memory_graph_tx, Arc<ServiceRegistry>)`
(embedder resolved from the `"embedder"` service; falls back to `NoopEmbedder`).
**Purpose:** query a domain corpus, list domains/manifests/sources, and drive
ingestion. **Notes:** the data plane is the user-level KnowledgeStore
(`knowledge_dir()`), separate from the project graph.

### `actors::rag_ingest` (ingestion library — not an actor)

```rust
pub struct GraphRagConfig { /* pipeline, knowledge domain, sources, chunking,
                               extraction rules, entity/relationship rules,
                               embedding + output config */ }
pub struct IngestContext {
    pub knowledge_tx: mpsc::Sender<MemoryGraphMessage>,
    pub memory_graph_tx: mpsc::Sender<MemoryGraphMessage>,
    pub embedder: std::sync::Arc<dyn Embedder>,
}
pub async fn ingest_graph_config(ctx: &IngestContext, manifest_path: &Path,
                                 project_root: Option<&Path>) -> Result<IngestReport>
pub fn resolve_domain(target_platform: &str) -> String
pub fn corpus_version_for(config: &GraphRagConfig) -> String   // 16-hex fingerprint
pub struct IngestReport { /* domain, chunks, entities, relationships, skipped sources */ }
pub struct SourceStatus { /* id, status, chunks, files, reason */ }
```
**Purpose:** parse the canonical `ingest.yaml` (`GraphRagConfig`) and execute a
full ingest: fetch sources → extract → chunk → batch-embed → store
chunks/entities/relationships → persist per-source status. See
[`rag.md`](rag.md).

### `actors::system_prompt` — `SystemPromptActor`

```rust
pub type ProjectToolCaller = std::sync::Arc<dyn Fn(String, serde_json::Value)
    -> futures::future::BoxFuture<'static, serde_json::Value> + Send + Sync>;
pub enum SystemPromptMessage {
    Initialize { project_tool_caller: ProjectToolCaller, reply_to: oneshot::Sender<Result<(), String>> },
    BuildPrefix { tools_hash: u64, tools: Vec<ToolInfo>, reply_to: oneshot::Sender<Vec<ChatMessageData>> },
    Invalidate,   // drop the cached prefix; next BuildPrefix rebuilds
}
```
**Purpose:** build the system-prompt prefix (with project context) and serve it
from cache while the tool set hash is unchanged.

### `actors::tool_providers` — `ToolRouterActor`

```rust
pub enum ToolRouterMessage {
    ListTools { reply_to: oneshot::Sender<Vec<ToolInfo>> },
    CallTool  { tool_name: String, args: Value, reply_to: oneshot::Sender<Result<Value, String>> },
}
pub fn ToolRouterActor::new(tool_registry, mcp_client_tx) -> Self
```
**Purpose:** generic tool dispatcher. Owns no tool definitions — everything is
registered into the shared `ToolRegistry` by the app composer; MCP server tools
are the one dynamic backend, queried per request. See also
`tool_providers::registry` (`ToolRegistry`).

### `actors::prompt_handler` — `PromptHandlerActor`

```rust
pub struct PromptContext { pub intent_name: Option<String>, pub project_context: Option<String>,
                           pub memories: Vec<String>, pub error_types: Vec<String>,
                           pub fix_strategies: Vec<String>, pub relevant_tools: Vec<String>,
                           pub build_state: Option<String> }
pub enum PromptHandlerMessage {
    HandlePrompt { query, intent_name, build_context, reply_to: oneshot::Sender<Result<String, String>> },
    HandlePromptWithTools { query, intent_name, build_context, tools: Vec<ToolInfo>,
                            reply_to: oneshot::Sender<Result<String, String>> },
}
```
**Purpose:** gather context from the memory graph, build a system prompt, and
drive the LLM (with optional tool-call lifecycle).
**Notes:** currently re-exported and kept as forward-looking API — the consumer
has not wired it into the running system yet.

## subsystems

### `subsystems::chat` — `ChatActor`

```rust
pub struct ChatMessageData { pub id: String, pub role: String, pub content: String,
                             pub timestamp: String,
                             pub widget: Option<serde_json::Value> }   // opaque widget JSON
pub struct ChatDialog { pub id: String, pub title: String,
                        pub messages: Vec<ChatMessageData>,
                        pub created_at: String, pub updated_at: String }
pub enum ChatMessage {
    GetActive   { reply_to: oneshot::Sender<Option<ChatDialog>> },
    GetHistory  { reply_to: oneshot::Sender<Vec<ChatDialog>> },
    Append      { chat_id: String, content: String, role: String,
                  widget: Option<serde_json::Value>, reply_to: Responder<ChatMessageData> },
    Clear       { chat_id: String, reply_to: Responder<()> },
    SetTitle    { chat_id: String, title: String, reply_to: Responder<()> },
    UpdateWidget{ widget_id: String, state: serde_json::Value, reply_to: Responder<()> },
}
pub fn ChatActor::new() -> Self
```
**Purpose:** chat dialog/message store for the extension UI; supports embedded
widgets (build-list, radio-group, checkbox-list, progress-bar) stored as opaque
JSON so the frontend drives rendering.

### `subsystems::graph` — `MemoryGraphActor`

The **sole data store** for the system: nodes, edges, vector embeddings, config,
bootstrap data, and transactional streams — all over SeleneDB's `GraphDb` via
GQL. See [`models::memory_graph`](#modelsmemory_graph) for the payload types.

```rust
pub enum MemoryGraphMessage {
    // Lifecycle
    Initialize         { data_dir: PathBuf, reply_to: Responder<()> },
    InitializeEmbedder { model_path: Option<PathBuf>, embedder: Option<Arc<dyn Embedder>>,
                         reply_to: Responder<()> },
    Sync               { /* snapshot + flush */ },
    // Nodes
    StoreAttrNode   { node: AttrNode, reply_to: Responder<AttrNode> },
    MergeAttrNode   { node: AttrNode, reply_to: Responder<AttrNode> },  // upsert by (type,name)
    GetAttrNode     { id: String, reply_to: Responder<AttrNode> },
    QueryAttrNodes  { node_type, subtype, name, limit, reply_to: Responder<Vec<AttrNode>> },
    UpdateNode      { id: String, updates: NodeUpdate, reply_to: Responder<AttrNode> },
    AtomicUpdateNode { id, property, expected_value, new_value, reply_to: Responder<AttrNode> },
    DeleteNode      { id: String, reply_to: Responder<()> },
    // Edges
    CreateRelationship { input: RelationshipInput, reply_to: Responder<GraphEdge> },
    GetRelationships   { node_id: String, reply_to: Responder<Vec<GraphEdge>> },
    DeleteRelationship { edge_id: String, reply_to: Responder<()> },
    Traverse           { start_id, options: TraversalOptions, reply_to: Responder<TraversalResult> },
    // Context / memory
    GetProjectContext  { project_root, reply_to: Responder<Option<...>> },
    SearchContext      { query, options: SearchOptions, reply_to: Responder<Vec<ContextSearchResult>> },
    AddMemory          { entry: MemoryEntry, reply_to: Responder<()> },
    Recall             { reply_to: Responder<Vec<MemoryEntry>> },
    // Config
    SetConfig { key, value, reply_to: Responder<()> },  GetConfig { key, reply_to: Responder<...> },
    BootstrapMcpConfig { reply_to: Responder<()> },     GetMcpConfig { reply_to: Responder<McpConfigFile> },
    BootstrapPlatforms { reply_to: Responder<()> },     GetPlatforms { reply_to: Responder<Vec<...>> },
    SeedIntents { reply_to: Responder<()> },
    // Transactions / bulk
    BatchGql { statements: Vec<String>, reply_to: Responder<Vec<StreamOpResult>> },
    OpenTransactionStream { reply_to: Responder<mpsc::Sender<TransactionRequest>> },
}
pub fn MemoryGraphActor::new() -> Self   // also impl Default
```
**Purpose:** all graph persistence; UUID↔`u64` ID mapping; GQL-first access;
`OpenTransactionStream` gives callers an atomic stream of `StreamOp`s committed
with `Commit`/`Rollback`.

### `subsystems::llm` — `LlmActor`

```rust
pub enum LlmModelRole { Default, Planning, Coding }   // selects planning/coding model
pub struct LlmConfig { pub api_url: String, pub api_key: String, pub model: String,
                       pub max_tokens: u32, pub coding_max_tokens: u32, pub temperature: f32,
                       pub strict_mode: bool, pub planning_model: String,
                       pub coding_model: String }     // Default available
pub enum LlmMessage {
    Complete { prompt: String, role: LlmModelRole, reply_to: Responder<String> },
    CompleteDefault { prompt: String, reply_to: Responder<String> },
    CompleteWithMessages { messages: Vec<ChatMessageData>, reply_to: Responder<String> },
    CompleteWithTools { messages: Vec<ChatMessageData>, tools: Vec<ToolInfo>,
                        reply_to: Responder<String> },
    Stream { prompt: String, reply_to: oneshot::Sender<Result<mpsc::Receiver<String>, ActorError>> },
    UpdateConfig { config: LlmConfig, reply_to: Responder<()> },
}
pub fn LlmActor::new(config: LlmConfig) -> Self
pub async fn complete_prompt(&self, prompt: &str, role: LlmModelRole) -> Result<String, ActorError>
```
**Purpose:** DeepSeek-compatible chat completions with planning/coding model
selection, streaming, and OpenAI-compatible `tools` arrays. Guards on a missing
API key.

### `subsystems::mcp` — `McpClientActor`

```rust
pub enum McpClientMessage {
    LoadConfigFromGraph { servers: Vec<McpServerConfig>, reply_to: Responder<()> },
    AddConfig { config: McpServerConfig, reply_to: Responder<()> },
    ConnectAll / Connect { server_name } / DisconnectAll / Disconnect { server_name },
    GetTools { server_name, reply_to: oneshot::Sender<Option<Vec<Tool>>> },
    ConnectedServers { reply_to: oneshot::Sender<Vec<String>> },
    GetConnectedServersWithTools { reply_to: oneshot::Sender<Vec<(String, Vec<Tool>)>> },
    GetServerDetails { reply_to: oneshot::Sender<Vec<McpServerDetail>> },
    CallTool { server_name, tool_name, arguments, reply_to: Responder<CallToolResult> },
    SetInternalTools { tools: Vec<Tool>, reply_to: Responder<()> },
    GetBuildSystemInfo { server_name, reply_to: oneshot::Sender<Option<BuildSystemInfo>> },
    GetBuildServers { reply_to: oneshot::Sender<Vec<(String, BuildSystemInfo)>> },
}
pub struct McpServerDetail { pub name, pub description, pub server_type, pub tool_count,
                             pub properties, pub build_type: Option<String> }
pub fn McpClientActor::new() -> Self
pub fn with_progress(progress_tx: Sender<ProgressMessage>) -> Self
```
**Purpose:** wrap `McpClientManager` behind message-passing; connect/disconnect
stdio or HTTP MCP servers, discover and call their tools, and expose a pseudo
"spire" server containing internal tools.

### `subsystems::tools` — `FileWatcherActor`, `ToolOrchestrator`

```rust
pub enum FileWatcherMessage {
    StartWatching { root: PathBuf, output: mpsc::Sender<FileChangeNotification>,
                    reply_to: oneshot::Sender<Result<(), String>> },
    StopWatching,            // stop watcher + debounce task (actor stays alive)
    Shutdown,                // stop everything, end the task
}
pub enum FileChangeNotification {
    InitialScan { root: PathBuf, files: Vec<FileInfo>, build_configs: Vec<(String, String)> },
    Batch { batch: FileEventBatch },
}
pub struct FileEventBatch { pub events: Vec<FileEventInfo>, pub batch_id: u64, pub timestamp: DateTime<Utc> }
pub struct FileEventInfo { pub path: PathBuf, pub kind: FileChangeKind }
pub enum FileChangeKind { Create, Modify, Remove, Rename, Other(String) }

pub enum ToolOrchestratorMessage {
    ExecuteTool { tool_name: String, parameters: HashMap<String, String>, reply_to: Responder<String> },
    ExecuteToolChain { tools: Vec<String>, parameters: HashMap<String, String>,
                       reply_to: Responder<Vec<String>> },
    ExecuteToolChainWithContext { tools: Vec<String>, error: BuildError, project_root: String,
                                  reply_to: Responder<Vec<String>> },
}
pub struct StepContext { /* build-error context + variable resolution */ }
pub fn ToolOrchestrator::new(memory_graph_tx, transport_tx, mcp_tx, llm_tx, tool_router_tx) -> Self
```
**Purpose:** `FileWatcherActor` produces an initial scan then debounced
(500 ms) change batches for the extension. `ToolOrchestrator` executes
graph-driven multi-step plans (falling back to the generic `ToolRouter` for
unregistered steps), resolving `{{variable}}` expressions against build-error
context.

## modules

Long-lived top-level `Actor`s (they implement `Actor`, not `ChildActor`)
registered under stable keys. Each exposes its own message enum and a `ListTools`
message for uniform capability discovery. All replies use `Responder<T>`
(`oneshot::Sender<Result<T, ActorError>>`).

### `modules::filesystem` — `FilesystemModule`

```rust
pub struct FilesystemModule;                          // static child actor
pub enum FilesystemMessage { /* read, write, list, delete, move, copy,
                                apply_patch, list_tools, … */ }
```
**Purpose:** file operations with parent-dir creation and a strict, context-
verified unified-diff `apply_patch` tool (apply-or-fail — never corrupts a file
with stale edits). Registered as `"filesystem"`.

### `modules::git` — `GitModule`

```rust
pub struct GitModule;
pub enum GitMessage { /* status, diff, log, branch, commit, … */ }
```
**Purpose:** read git state and drive simple commit workflows. Registered as
`"git"`.

### `modules::process` — `ProcessModule`

```rust
pub struct ProcessModule;
pub enum ProcessMessage { /* spawn, kill, list, monitor, … */ }
```
**Purpose:** subprocess lifecycle for long-running or monitored processes.
Registered as `"process"`.

### `modules::search` — `SearchModule`

```rust
pub struct SearchModule;
pub enum SearchMessage { /* file/content search, list_tools, … */ }
```
**Purpose:** filesystem and content search (grep-style). Registered as
`"search"`.

### `modules::terminal` — `TerminalModule`

```rust
pub struct TerminalModule;
pub enum TerminalMessage { /* execute command, capture output, … */ }
```
**Purpose:** execute commands and capture combined output for the terminal tool.
Registered as `"terminal"`.

**Construction note:** consumers spawn these with
`spawn_module(Module::new())` (a `ChildActor` → top-level `Actor` adapter) and
register the resulting sender under the module key.

## models

### `models::embedding` — `Embedder` trait, `Embedding`

```rust
pub struct Embedding {
    pub vector: Vec<f32>,            // L2-normalized, 384-d for all-MiniLM-L6-v2
    pub text: String,
    pub text_hash: String,           // MD5, for caching/dedup
    pub token_count: usize,          // whitespace split
    pub dimensions: usize,
    pub model_name: String,
    pub generated_at: DateTime<Utc>,
}
impl Embedding { pub fn new(vector: Vec<f32>, text: &str, model_name: &str) -> Self }

#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> anyhow::Result<Embedding>;
    async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Embedding>>;
    fn dimensions(&self) -> usize;
}
```
**Purpose:** the `Embedder` abstraction is the seam between the RAG pipeline and
any embedding backend; `Embedding` carries the vector plus provenance metadata.
**Notes:** mirrors the TypeScript `IEmbedder` contract from the original `spire`
project.

### `models::memory_graph`

The graph data contract used by `MemoryGraphActor` (and the RAG store):

```rust
// Transaction stream (atomic ops, committed via StreamOp::Commit)
pub enum StreamOp {
    StoreNode(AttrNode), StoreNodeWithEmbedding { node, embedding_vector },
    UpdateNode { id, updates: NodeUpdate }, DeleteNode(String),
    CreateRelationship(RelationshipInput), DeleteRelationship(String),
    SetConfig { key, value }, RawGql(String),
    MergeNode(AttrNode), MergeRelationship(RelationshipInput),
    Commit, Rollback,
}
pub enum StreamOpResult { NodeStored(AttrNode), NodeUpdated(AttrNode), NodeDeleted,
                          RelationshipCreated(GraphEdge), RelationshipDeleted,
                          ConfigSet, RawGql(Option<serde_json::Value>) }
pub struct TransactionRequest { pub operation: StreamOp,
                                pub reply_to: oneshot::Sender<Result<StreamOpResult, String>> }

pub enum SchemaError { DuplicateNode { type_name, name }, NodeNotFound { id },
                       AcyclicDependencyViolation { from, to } }

pub struct AttrNode { pub id: String, pub node_type: String, pub subtype: Option<String>,
                      pub name: String, pub description: Option<String>,
                      pub properties: HashMap<String, serde_json::Value>,
                      pub embedding_id: Option<String>,
                      pub created_at: DateTime<Utc>, pub updated_at: DateTime<Utc>,
                      pub version: u32 }
// + typed accessors: get, u32_prop, f64_prop, bool_prop, str_prop,
//                    str_array_prop, is(node_type), diagnostic(), …
pub struct NodeUpdate { /* partial update envelope */ }
pub enum RelationshipType { /* typed relation kinds + Custom(String) */ }
pub struct RelationshipInput { /* from, to, rel_type, properties */ }
pub struct GraphEdge { /* id, from, to, edge_type, properties */ }
pub struct SearchOptions { /* filters, limit, embedding-based search flags */ }
pub struct ContextSearchResult { /* node + score + provenance */ }
pub struct ScoredNode { pub node: AttrNode, pub score: f32, pub source: RetrievalSource }
pub enum RetrievalSource { Vector, Gql, /* … */ }
pub struct TraversalOptions { pub direction: TraversalDirection, pub max_depth: usize, /* … */ }
pub enum TraversalDirection { Outgoing, Incoming, Both }
pub struct TraversalResult { pub paths: Vec<TraversalPath> }
pub struct MemoryMetadata { /* memory provenance */ }
pub struct MemoryEntry { /* a remembered fact with metadata */ }
pub struct McpServerConfigEntry { pub name, pub command, pub args, pub env, pub url,
                                  pub headers, pub autostart }
pub struct McpConfigFile { pub servers: Vec<McpServerConfigEntry> }
```
The module also carries the **build/planning domain model** consumed by
`spire-code`'s build & planning subsystems: `BuildState`, `BuildContext`,
`BuildResult`, `BuildError`, `SystemBuildResult`, `BuildStartResult`, `FixPlan`,
`FixStrategy`, `ScoredFix`, `AnnotatedError`, `Intent`, `ErrorType`,
`PlanStatus`, `PlanStepData`, `PlanStepEntry`, `PlanStatusResult`, `Tool`,
`ToolCategory`.

### `models::analysis`

```rust
pub struct CodeAnalysisRequest { /* source text + options */ }
pub struct CodeAnalysis { /* symbols, complexity, … */ }
pub struct ComplexityScore { /* cyclomatic etc. */ }
pub struct SymbolInfo { /* name, kind, span, … */ }
pub enum SymbolKind { Function, Class, Struct, Enum, Interface, /* … */ }
pub struct GraphSymbol { /* … */ }
impl GraphSymbol { pub fn from_attr_node(attr: &AttrNode) -> Option<Self> }
pub struct SearchResult { /* … */ }
pub struct SearchRequest { /* … */ }
```
**Purpose:** code-analysis result types that round-trip with `AttrNode` nodes in
the knowledge graph (e.g. `AstFunction`/`AstClass` interface nodes used by
`RagActor::FindInterfaces`).

## embedder

```rust
pub struct CandleEmbedder;                       // all-MiniLM-L6-v2 via Candle
impl CandleEmbedder { pub fn new() -> anyhow::Result<Self> }
pub fn create_embedder() -> anyhow::Result<CandleEmbedder>

pub struct NoopEmbedder;                         // fail-loud degraded mode
impl Embedder for NoopEmbedder { /* embed/embed_batch always Err; dimensions() == 0 */ }
```
**Purpose:** `CandleEmbedder` runs the sentence-transformer locally (384-d,
L2-normalized). `NoopEmbedder` **never returns zero vectors** — it fails loudly
so RAG ingest/search surfaces a clear error instead of silently degrading.
**Notes:** the embedder is shared as `Arc<dyn Embedder>`; to put it in the
`ServiceRegistry`, wrap it in `actors::rag::EmbedderService` (sized wrapper, so
`Any`-downcast works) and register under `"embedder"`.

## mcp

### `mcp::client` — `McpClientManager`

```rust
pub struct McpClientManager;
impl McpClientManager {
    pub fn new() -> Self;
    pub fn load_config_from_entries(&mut self, servers: Vec<McpServerConfig>);  // replaces configs
    // + connect/disconnect, list_tools, call_tool, build-system introspection
}
impl Default for McpClientManager { /* new() */ }

#[serde(tag = "type")]
pub enum TransportConfig {
    Stdio { command: String, #[serde(default)] args: Vec<String>,
            #[serde(default)] env: HashMap<String, String> },
    Http  { url: String, #[serde(default)] headers: HashMap<String, String> },
}
pub struct McpServerConfig {
    pub name: String,
    pub transport: TransportConfig,
    #[serde(default = "default_autostart")] pub autostart: bool,
    #[serde(default)] pub build_type: Option<String>,
}
pub struct BuildSystemInfo { pub name: String, pub build_systems: Vec<String>,
                             pub build_type: String, pub capabilities: Vec<String>,
                             pub analyzer_tool: Option<String>,
                             pub config_files: Vec<String> }
```
**Purpose:** connect to external MCP servers (stdio subprocess or HTTP), discover
their tools and capabilities, and call tools. It is **purely a client** — it does
not host an MCP server. `BuildSystemInfo` is the self-description returned by
build MCP servers via their `_build_system` tool.
**Notes:** production code drives the manager through `McpClientActor`
(`subsystems::mcp`), not directly. The public config types are also used by
`spire-code` for wiring.

## config

User-level (global) configuration under `~/.spire` (overridable via
`$SPIRE_CONFIG_DIR`). LLM settings live in `llm-config.json`; the shared RAG
corpus lives in the knowledge dir.

```rust
pub fn config_dir() -> PathBuf;                       // $SPIRE_CONFIG_DIR | ~/.spire
pub fn llm_config_path() -> PathBuf;                  // config_dir()/llm-config.json
pub fn knowledge_dir() -> PathBuf;                    // $SPIRE_KNOWLEDGE_DIR | ~/.spire/knowledge
pub fn load_global_llm_config() -> LlmConfig;         // deepseek.* keys → LlmConfig
pub fn get_global_llm_config_key(key: &str) -> Option<String>;
pub fn set_global_llm_config_key(key: &str, value: &str) -> Result<LlmConfig, String>; // atomic write
pub fn global_config_json() -> Value;                 // settings payload for the UI
pub const DEEPSEEK_KEYS: [&str; 5];                   // deepseek.api_key, .model, .api_url,
                                                      // .planning_model, .coding_model
pub const WEB_SEARCH_KEYS: [&str; 1];                 // tavily.api_key
```
**Purpose:** one shared settings store for LLM + web-search keys, readable by any
actor without message traffic. `set_global_llm_config_key` writes via temp-file +
rename (atomic).
**Notes:** tests use the `SPIRE_CONFIG_DIR` / `SPIRE_KNOWLEDGE_DIR` overrides to
avoid touching the user's real `~/.spire`.

## analyzer

Local project analysis (no MCP servers involved):

```rust
// scanner — walk a directory tree, respecting .gitignore
pub fn scan_directory(root: &Path, no_ignore: bool) -> Vec<FileInfo>;
pub fn discover_build_files(root: &Path, no_ignore: bool) -> Vec<(String, String)>; // path, type

// tree_builder — assemble the flat file list into a hierarchical tree
pub fn build_file_tree(root: &Path, no_ignore: bool) -> DirectoryNode;

// models
pub struct FileInfo { pub path: String, pub relative_path: String, pub extension: String,
                      pub size: u64, pub is_dir: bool, pub is_symlink: bool }
pub struct DirectoryNode { pub name: String, pub path: String, pub role: String,
                           pub directories: Vec<DirectoryNode>, pub files: Vec<FileNode>,
                           pub total_file_count: usize, pub total_lines: usize }
pub struct FileNode { pub name: String, pub path: String, pub extension: String,
                      pub language: String, pub size: u64,
                      pub lines_estimated: usize, pub role: String }
```
**Purpose:** filesystem scanning and semantic file-tree building (language
detection, role classification, line estimation) used by `FileWatcherActor` and
`spire-code`'s `ProjectAnalyzerActor`.
**Notes:** build-system-specific parsing is **not** done here — it lives in
external MCP servers (mcp-cargo, mcp-node, …). `analyzer::models` re-exports the
`BuildMetadata` contract from `build_types`.

## graph

`crate::graph::GraphDb` — SeleneDB-backed graph with WAL + snapshot persistence:

```rust
pub struct GraphDb;                                    // Send + Sync (unsafe impls)
impl GraphDb {
    pub fn new_in_memory() -> anyhow::Result<Self>;
    pub fn new_with_wal(wal_path: &Path) -> anyhow::Result<Self>;
    pub fn recover(data_dir: &Path, graph_id: u64) -> anyhow::Result<Self>; // snapshot + WAL
    pub fn latest_snapshot_sequence(data_dir: &Path) -> anyhow::Result<u64>;
    // node/edge CRUD, vector search, GQL query/write, snapshots
    pub fn execute_gql_query(&self, query: &str) -> anyhow::Result<BindingTable>;
    pub fn execute_gql_write(&self, statement: &str) -> anyhow::Result<()>;
    pub fn begin_transaction(&self) -> Result<GraphDbTransaction>;
    // …
}
pub type GraphDbTransaction = WriteTxn<'static>;       // commits on drop (RAII)
pub fn to_db_string(s: &str) -> Value;                 // SeleneDB string wrapper
```
**Purpose:** low-level storage for `MemoryGraphActor`. In production, actors go
through GQL via `MemoryGraphActor` — do not use the low-level `SharedGraph` API
outside `graph.rs`.
**Notes:** contains `unsafe impl Send/Sync` (the wrapper is thread-safe by
construction); its `#[cfg(test)]` suite exercises CRUD, GQL, vector search, and
WAL recovery.

## platform

Cross-compilation platform definitions and cross-file generation (the typed
`Platform` struct lives in `build_types`; the methods in `platform.rs`):

```rust
impl Platform {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Platform>;
    pub fn load_directory(dir: impl AsRef<Path>) -> anyhow::Result<Vec<Platform>>;
    pub fn sysroot_ok(&self) -> (bool, String);
    pub fn default_platform_dir() -> PathBuf;           // ~/.spire/platforms
    pub fn from_registry(id: &str) -> Option<Platform>;
    pub fn cargo_config(&self) -> Option<String>;
    pub fn meson_cross_file(&self) -> Option<String>;
}
pub struct CrossSpec;                                   // generated cargo/meson specs
impl CrossSpec { pub fn for_platform(id: &str) -> Option<CrossSpec>; }
```
**Purpose:** represent and generate cross-compilation configuration (Cargo
`.cargo/config` snippets, Meson cross files) for targets such as `rpi5` /
`rock3c`, consumed by `spire-code`'s build actors.
**Notes:** honors the `SPIRE_PLATFORM_DIR` env override (used by tests/CI).

## transport

`transport::socket` — JSON-RPC 2.0 over TCP to the VS Code extension:

```rust
pub struct TransportActor;
impl TransportActor { pub fn new() -> Self }           // Default too
pub enum TransportMessage {
    Bind   { reply_to: oneshot::Sender<Result<u16, String>> },   // loopback port
    Accept { reply_to: oneshot::Sender<Result<(), String>> },
    CallExtension { method: String, params: Value, reply_to: oneshot::Sender<Result<Value, String>> },
    SendResponse / SendError / SendNotification { … },
    SetRequestHandler { handler_tx } / SetNotificationHandler { notification_tx } / SetSelfTx { self_tx },
    SocketLine { line: String } / RemovePending { id },
}
pub struct IncomingNotification { pub method: String, pub params: serde_json::Value }
```
**Purpose:** bind a loopback port, accept the extension's connection, forward
extension requests to a registered handler, and call extension methods
(`CallExtension`) with timeout + pending-request cleanup.
**Notes:** the pending-response await runs in a spawned task so the reader loop
never deadlocks.

## build_types

The **shared contract** between spire-core and the MCP build servers — types
serialized across the boundary so the two sides cannot diverge:

`BuildMetadata` (build_system, project_type, domains, targets), `BuildScript`,
`BuildTarget`, `Dependency`, `DomainEditability`, `Feature`,
`McpServerCapability`, `ProjectDomain`, `WorkspaceMember`, plus the platform
models `Platform`, `PlatformArchitecture`, `PlatformToolchain`,
`PlatformSysroot`, and helpers like `all_config_file_names()`. Re-exported at
`spire_core::analyzer::models::*` for consumers.





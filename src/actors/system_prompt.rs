// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! SystemPromptActor — assembles and caches the system prompt prefix for DeepSeek.
//!
//! The conversation prefix is 4 system messages designed to maximize DeepSeek prompt caching:
//!   [0] Base identity + tool-use instructions (static)
//!   [1] Available tools list (changes when MCP servers connect/disconnect)
//!   [2] Project overview from project/getOverview (changes on project switch)
//!   [3] Project architecture from project/getArchitecture (changes on project switch)
//!
//! Cache invalidation:
//! - Caller provides a u64 tools_hash with BuildPrefix — if it matches the cached hash,
//!   the actor serves cached messages without re-fetching project context.
//! - Invalidate forces rebuild on next BuildPrefix call.

use async_trait::async_trait;
use std::hash::{Hash, Hasher};
use tokio::sync::oneshot;
use tracing::{debug, info};

use crate::actors::{Actor, ToolInfo};
use crate::subsystems::chat::chat::ChatMessageData;

/// Type-erased project-tool caller injected by the composer (spire-code).
///
/// The system-prompt actor stays generic — it never sees the project actor's
/// message type; the composer wraps `ProjectQueryMessage::CallTool` in a
/// closure.
pub type ProjectToolCaller = std::sync::Arc<
    dyn Fn(String, serde_json::Value) -> futures::future::BoxFuture<'static, serde_json::Value>
        + Send
        + Sync,
>;

// ============================================================================
// Messages
// ============================================================================

pub enum SystemPromptMessage {
    /// Initialize the actor with the project-tool caller (composer-injected).
    Initialize {
        project_tool_caller: ProjectToolCaller,
        reply_to: oneshot::Sender<Result<(), String>>,
    },
    /// Build the prefix (or serve from cache if tools_hash matches).
    BuildPrefix {
        tools_hash: u64,
        tools: Vec<ToolInfo>,
        reply_to: oneshot::Sender<Vec<ChatMessageData>>,
    },
    /// Force cache invalidation on next BuildPrefix.
    Invalidate,
}

// ============================================================================
// Constants
// ============================================================================

const BASE_SYSTEM_PROMPT: &str = "You are an AI coding assistant for the Spire editor. \
You have access to tools that can read, analyze, build, search, and modify the project. \
When the user gives you a request, use your available tools to fulfill it rather than \
just describing what you would do. Always call the appropriate tool to carry out the \
user's request.

IMPORTANT — Project Query Tools:
- The project has ALREADY been fully scanned. All project structure information is \
stored in the knowledge graph and is accessible via the project query tools listed below.
- Do NOT use the terminal (bash, find, grep, etc.) to explore or discover the project \
structure. Instead, use the dedicated project query tools.
- Use `project/getOverview` for a high-level summary (languages, build systems, file counts).
- Use `project/getFileTree` to explore the directory/file tree with semantic annotations.
- Use `project/getArchitecture` for the high-level architectural overview (modules, roles).
- Use `project/getBuildConfig` for parsed build configuration and scripts.
- Use `project/getDependencies` for the dependency graph (external + internal).
- Use `project/getEntryPoints` for main entry points of the project.
- Use `project/searchFiles` to find files by name, language, role, or path pattern.
- Use `project/getFileDetails` for detailed metadata about a specific file.
- If you need to read a specific file's contents, use the workspace file tools — but for \
understanding the project layout, ALWAYS use the project query tools first.

IMPORTANT — Build Tool Usage:
- When the user asks you to build, compile, compile and run, or run the project, \
you MUST use the `project_build` tool.
- The `project_build` tool automatically discovers all build systems in the project \
(Cargo, npm, pnpm, yarn, etc.) and builds them in parallel.
- Do NOT try to run build commands manually via the terminal — use `project_build` instead.";

// ============================================================================
// Helpers
// ============================================================================

/// Sanitize a tool name for display in the system prompt.
///
/// The LLM API (DeepSeek/OpenAI) requires function names to match
/// `^[a-zA-Z0-9_-]+$`, so the LlmActor sanitizes tool names by replacing
/// invalid characters (like `/`) with underscores. We must show the same
/// sanitized names in the system prompt so the LLM sees consistent names
/// between the tool list and the API function definitions.
fn sanitize_tool_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ============================================================================
// Formatters
// ============================================================================

fn format_tool_list(tools: &[ToolInfo]) -> String {
    if tools.is_empty() {
        return "No tools are currently available.".to_string();
    }

    // Categorize tools by prefix
    let mut project_tools = Vec::new();
    let mut workspace_tools = Vec::new();
    let mut document_tools = Vec::new();
    let mut diagnostics_tools = Vec::new();
    let mut git_tools = Vec::new();
    let mut symbols_tools = Vec::new();
    let mut mcp_tools = Vec::new();
    let mut other_tools = Vec::new();

    for t in tools {
        if t.name.starts_with("project/") {
            project_tools.push(t);
        } else if t.name.starts_with("workspace/") {
            workspace_tools.push(t);
        } else if t.name.starts_with("document/") {
            document_tools.push(t);
        } else if t.name.starts_with("diagnostics/") {
            diagnostics_tools.push(t);
        } else if t.name.starts_with("git/") {
            git_tools.push(t);
        } else if t.name.starts_with("symbols/") {
            symbols_tools.push(t);
        } else if t.name.contains('/') {
            // Tools with a slash prefix that aren't in the above categories
            // are likely MCP server tools (e.g. "mcp-cargo/build")
            mcp_tools.push(t);
        } else {
            other_tools.push(t);
        }
    }

    let mut s = String::from(
        "You have access to the following tools. Use them to fulfill the user's request.\n\n",
    );

    // ── Project tools ──
    if !project_tools.is_empty() {
        s.push_str("=== Project Tools ===\n");
        s.push_str("Use these to query and build the project.\n");
        for t in &project_tools {
            s.push_str(&format!(
                "  - `{}`: {}\n",
                sanitize_tool_name(&t.name),
                t.description
            ));
        }
        s.push('\n');
    }

    // ── Workspace tools ──
    if !workspace_tools.is_empty() {
        s.push_str("=== Workspace Tools ===\n");
        s.push_str("Use these to read, search, and manage files in the workspace.\n");
        for t in &workspace_tools {
            s.push_str(&format!(
                "  - `{}`: {}\n",
                sanitize_tool_name(&t.name),
                t.description
            ));
        }
        s.push('\n');
    }

    // ── Document tools ──
    if !document_tools.is_empty() {
        s.push_str("=== Document Tools ===\n");
        s.push_str("Use these to read and edit open documents.\n");
        for t in &document_tools {
            s.push_str(&format!(
                "  - `{}`: {}\n",
                sanitize_tool_name(&t.name),
                t.description
            ));
        }
        s.push('\n');
    }

    // ── Diagnostics tools ──
    if !diagnostics_tools.is_empty() {
        s.push_str("=== Diagnostics Tools ===\n");
        s.push_str("Use these to check for errors and warnings in the project.\n");
        for t in &diagnostics_tools {
            s.push_str(&format!(
                "  - `{}`: {}\n",
                sanitize_tool_name(&t.name),
                t.description
            ));
        }
        s.push('\n');
    }

    // ── Git tools ──
    if !git_tools.is_empty() {
        s.push_str("=== Git Tools ===\n");
        s.push_str("Use these for version control operations.\n");
        for t in &git_tools {
            s.push_str(&format!(
                "  - `{}`: {}\n",
                sanitize_tool_name(&t.name),
                t.description
            ));
        }
        s.push('\n');
    }

    // ── Symbol tools ──
    if !symbols_tools.is_empty() {
        s.push_str("=== Symbol Tools ===\n");
        s.push_str("Use these to find and navigate code symbols.\n");
        for t in &symbols_tools {
            s.push_str(&format!(
                "  - `{}`: {}\n",
                sanitize_tool_name(&t.name),
                t.description
            ));
        }
        s.push('\n');
    }

    // ── MCP tools ──
    if !mcp_tools.is_empty() {
        s.push_str("=== External Tools (MCP Servers) ===\n");
        s.push_str("These tools are provided by connected MCP servers.\n");
        for t in &mcp_tools {
            s.push_str(&format!(
                "  - `{}`: {}\n",
                sanitize_tool_name(&t.name),
                t.description
            ));
        }
        s.push('\n');
    }

    // ── Other tools ──
    if !other_tools.is_empty() {
        s.push_str("=== Other Tools ===\n");
        for t in &other_tools {
            s.push_str(&format!(
                "  - `{}`: {}\n",
                sanitize_tool_name(&t.name),
                t.description
            ));
        }
        s.push('\n');
    }

    // ── Usage guidance ──
    s.push_str(
        "=== Usage Guidance ===\n\
         - When the user asks you to BUILD, COMPILE, or RUN the project, \
         use the `project_build` tool.\n\
         - When the user asks you to READ a file, use the workspace file tools.\n\
         - When the user asks you to SEARCH the codebase, use the workspace search tools.\n\
         - When the user asks you about project structure, use the project query tools.\n\
         - When the user asks you to check for ERRORS, use the diagnostics tools.\n\
         - When the user asks you about GIT history or changes, use the git tools.\n",
    );

    s
}

fn format_overview(value: &serde_json::Value) -> String {
    if let Some(err) = value.get("error").and_then(|v| v.as_str()) {
        return format!("Project context: unavailable (error: {})", err);
    }
    let name = value
        .get("projectName")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let files = value
        .get("totalFiles")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let dirs = value.get("totalDirs").and_then(|v| v.as_u64()).unwrap_or(0);
    let lines = value
        .get("totalLines")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let mut s = format!(
        "Project Overview\nProject: {}\nFiles: {}, Directories: {}, Total lines: ~{}\n",
        name, files, dirs, lines
    );

    if let Some(langs) = value.get("languages").and_then(|v| v.as_array()) {
        if !langs.is_empty() {
            s.push_str("\nLanguages:\n");
            for lang in langs {
                let l = lang.get("language").and_then(|v| v.as_str()).unwrap_or("?");
                let c = lang.get("fileCount").and_then(|v| v.as_u64()).unwrap_or(0);
                s.push_str(&format!("  - {}: {} files\n", l, c));
            }
        }
    }

    if let Some(bss) = value.get("buildSystems").and_then(|v| v.as_array()) {
        if !bss.is_empty() {
            s.push_str("\nBuild Systems:\n");
            for bs in bss {
                let n = bs.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                let t = bs.get("type").and_then(|v| v.as_str()).unwrap_or("?");
                s.push_str(&format!("  - {} ({})\n", n, t));
            }
            s.push_str("\nTo build this project, use the `project_build` tool.\n");
        }
    }

    s
}

fn format_architecture(value: &serde_json::Value) -> String {
    if let Some(err) = value.get("error").and_then(|v| v.as_str()) {
        return format!("Project architecture: unavailable (error: {})", err);
    }

    let mut s = String::from("Project Architecture\n");

    if let Some(modules) = value.get("modules").and_then(|v| v.as_array()) {
        if !modules.is_empty() {
            s.push_str("\nKey Modules:\n");
            for m in modules {
                let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("?");
                let desc = m.get("description").and_then(|v| v.as_str()).unwrap_or("");
                let count = m.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
                s.push_str(&format!(
                    "  - **{}**: {} ({} directories)\n",
                    role, desc, count
                ));
            }
        }
    }

    if let Some(eps) = value.get("entryPoints").and_then(|v| v.as_array()) {
        if !eps.is_empty() {
            s.push_str("\nEntry Points:\n");
            for ep in eps {
                if let Some(path) = ep.as_str() {
                    s.push_str(&format!("  - {}\n", path));
                }
            }
        }
    }

    s
}

// ============================================================================
// Actor
// ============================================================================

pub struct SystemPromptActor {
    /// Channel to ProjectQueryActor (composer-injected, type-erased).
    project_tool_caller: Option<ProjectToolCaller>,
    /// Cached prefix messages (4 system messages).
    cached_messages: Vec<ChatMessageData>,
    /// Cached tools hash — rebuild only when this changes.
    cached_tools_hash: u64,
    /// Whether the cache is valid.
    cache_valid: bool,
}

impl SystemPromptActor {
    pub fn new() -> Self {
        Self {
            project_tool_caller: None,
            cached_messages: Vec::new(),
            cached_tools_hash: 0,
            cache_valid: false,
        }
    }

    /// Compute a hash from tool definitions to detect changes.
    fn tools_hash(tools: &[ToolInfo]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for t in tools {
            t.name.hash(&mut hasher);
            t.description.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Build the 4-message prefix and cache it.
    async fn build_prefix(&mut self, tools: &[ToolInfo]) -> Vec<ChatMessageData> {
        let now = chrono::Utc::now().to_rfc3339();
        let mut messages = Vec::with_capacity(4);

        // [0] Base system prompt
        messages.push(ChatMessageData {
            id: "system-base".to_string(),
            role: "system".to_string(),
            content: BASE_SYSTEM_PROMPT.to_string(),
            timestamp: now.clone(),
            widget: None,
        });

        // [1] Tools list
        messages.push(ChatMessageData {
            id: "system-tools".to_string(),
            role: "system".to_string(),
            content: format_tool_list(tools),
            timestamp: now.clone(),
            widget: None,
        });

        // [2] Project overview
        let overview = self
            .call_project_tool("project/getOverview", serde_json::json!({}))
            .await;
        messages.push(ChatMessageData {
            id: "system-overview".to_string(),
            role: "system".to_string(),
            content: format_overview(&overview),
            timestamp: now.clone(),
            widget: None,
        });

        // [3] Project architecture
        let architecture = self
            .call_project_tool("project/getArchitecture", serde_json::json!({}))
            .await;
        messages.push(ChatMessageData {
            id: "system-architecture".to_string(),
            role: "system".to_string(),
            content: format_architecture(&architecture),
            timestamp: now,
            widget: None,
        });

        // Cache
        self.cached_messages = messages.clone();
        self.cached_tools_hash = Self::tools_hash(tools);
        self.cache_valid = true;

        messages
    }

    /// Fire a synchronous call to a ProjectQuery tool and await the response.
    async fn call_project_tool(&self, tool: &str, args: serde_json::Value) -> serde_json::Value {
        let caller = match &self.project_tool_caller {
            Some(caller) => caller,
            None => return serde_json::json!({"error": "ProjectQuery channel not set"}),
        };
        caller(tool.to_string(), args).await
    }

    /// Handle incoming messages.
    pub async fn handle_message(&mut self, msg: SystemPromptMessage) {
        match msg {
            SystemPromptMessage::Initialize {
                project_tool_caller,
                reply_to,
            } => {
                self.project_tool_caller = Some(project_tool_caller);
                info!("SystemPromptActor: initialized");
                let _ = reply_to.send(Ok(()));
            }
            SystemPromptMessage::BuildPrefix {
                tools_hash,
                tools,
                reply_to,
            } => {
                // If cache is valid and hash matches, serve cached messages
                if self.cache_valid && tools_hash == self.cached_tools_hash {
                    let _ = reply_to.send(self.cached_messages.clone());
                    debug!("SystemPromptActor: cache HIT (tools_hash={})", tools_hash);
                    return;
                }

                // Cache miss or invalidated — rebuild
                debug!(
                    "SystemPromptActor: cache MISS (valid={}, hash_old={}, hash_new={})",
                    self.cache_valid, self.cached_tools_hash, tools_hash
                );
                let messages = self.build_prefix(&tools).await;
                let _ = reply_to.send(messages);
            }
            SystemPromptMessage::Invalidate => {
                self.cache_valid = false;
                info!("SystemPromptActor: cache invalidated");
            }
        }
    }
}

impl Default for SystemPromptActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for SystemPromptActor {
    type Message = SystemPromptMessage;

    async fn handle(&mut self, msg: Self::Message) {
        self.handle_message(msg).await;
    }
}

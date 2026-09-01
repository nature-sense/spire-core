// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! ToolOrchestrator actor — executes tools and tool chains for the build-fix loop.
//!
//! Fully graph-driven: reads StepDefinition and ToolProvider nodes from the
//! graph to determine what tool to call, what arguments to pass, and which
//! provider (VSC extension, MCP server, LLM, project-meta) to route through.
//!
//! Step execution is chained: each step's output_key feeds into subsequent
//! steps via StepContext, and dependency resolution validates the chain order.

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::subsystems::llm::llm::LlmMessage;
use crate::subsystems::mcp::mcp_client::McpClientMessage;
use crate::subsystems::graph::memory_graph::MemoryGraphMessage;
use crate::actors::Actor;
use crate::models::memory_graph::BuildError;
use crate::transport::socket::TransportMessage;

// ============================================================================
// StepContext — carries error context and previous step outputs for chaining
// ============================================================================

/// Context passed through a tool chain execution. Each step can read from
/// the error context and write to the output store for subsequent steps.
#[derive(Debug, Clone, Default)]
pub struct StepContext {
    /// The original build error
    pub error: Option<BuildError>,
    /// The project root path
    pub project_root: String,
    /// Outputs from previous steps, keyed by output_key
    pub step_outputs: HashMap<String, serde_json::Value>,
    /// The build system type that produced the error (e.g. "Cargo", "npm")
    pub build_type: String,
}

impl StepContext {
    pub fn new(error: &BuildError, project_root: &str) -> Self {
        Self {
            error: Some(error.clone()),
            project_root: project_root.to_string(),
            step_outputs: HashMap::new(),
            build_type: error.build_type.clone().unwrap_or_default(),
        }
    }

    /// Register the output of a completed step.
    pub fn set_output(&mut self, key: &str, value: serde_json::Value) {
        self.step_outputs.insert(key.to_string(), value);
    }

    /// Get a value by dot-path, with variable substitution support.
    /// Supports: $error.file, $error.line, $error.column, $error.error_text,
    /// $error.build_type, $project_root, $step.{key}, $step.{key}.{field}
    pub fn resolve_variable(&self, expr: &str) -> Option<serde_json::Value> {
        // Remove the $ prefix
        let body = expr.strip_prefix('$')?;

        // Try $error.*
        if let Some(rest) = body.strip_prefix("error.") {
            return self.resolve_error_field(rest);
        }

        // Try $project_root
        if body == "project_root" {
            return Some(serde_json::Value::String(self.project_root.clone()));
        }

        // Try $step.{output_key}.{nested.field}
        if let Some(rest) = body.strip_prefix("step.") {
            return self.resolve_step_output(rest);
        }

        None
    }

    fn resolve_error_field(&self, field: &str) -> Option<serde_json::Value> {
        let error = self.error.as_ref()?;
        match field {
            "file" => error
                .file
                .as_ref()
                .map(|s| serde_json::Value::String(s.clone())),
            "line" => error
                .line
                .map(|n| serde_json::Value::Number(serde_json::Number::from(n))),
            "column" => error
                .column
                .map(|n| serde_json::Value::Number(serde_json::Number::from(n))),
            "error_text" => Some(serde_json::Value::String(error.error_text.clone())),
            "error_type" => error
                .error_type
                .as_ref()
                .map(|s| serde_json::Value::String(s.clone())),
            "exit_code" => error
                .exit_code
                .map(|c| serde_json::Value::Number(serde_json::Number::from(c))),
            "build_type" => error
                .build_type
                .as_ref()
                .map(|s| serde_json::Value::String(s.clone())),
            _ => None,
        }
    }

    fn resolve_step_output(&self, path: &str) -> Option<serde_json::Value> {
        // path is like "read_error_context.file_context" or just "analysis" or "search_results[0]"
        // Split on '.' but NOT inside brackets
        let output_key = if let Some(bracket_pos) = path.find('[') {
            // Handle "search_results[0]" — extract "search_results" as the key
            let key = &path[..bracket_pos];
            let rest = &path[bracket_pos..];
            let value = self.step_outputs.get(key)?;
            // rest is like "[0]" or "[0].field"
            self.resolve_indexed(value, rest)
        } else if let Some(dot_pos) = path.find('.') {
            let key = &path[..dot_pos];
            let value = self.step_outputs.get(key)?;
            let field_path = &path[dot_pos + 1..];
            self.resolve_nested(value, field_path)
        } else {
            self.step_outputs.get(path).cloned()
        };
        output_key
    }

    /// Resolve array index access like "[0]" or "[0].field"
    fn resolve_indexed(
        &self,
        value: &serde_json::Value,
        bracket_path: &str,
    ) -> Option<serde_json::Value> {
        // bracket_path is like "[0]" or "[0].field"
        let close_bracket = bracket_path.find(']')?;
        let idx_str = &bracket_path[1..close_bracket];
        let idx: usize = idx_str.parse().ok()?;
        let arr = value.as_array()?;
        let item = arr.get(idx)?;
        let rest = &bracket_path[close_bracket + 1..];
        if rest.is_empty() {
            return Some(item.clone());
        }
        // Skip leading '.' if present
        let field_path = rest.strip_prefix('.').unwrap_or(rest);
        self.resolve_nested(item, field_path)
    }

    fn resolve_nested(&self, value: &serde_json::Value, path: &str) -> Option<serde_json::Value> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = value;
        for part in parts {
            // Array index access: search_results[0]
            if let Ok(idx) = part.parse::<usize>() {
                current = current.get(idx)?;
            } else {
                current = current.get(part)?;
            }
        }
        Some(current.clone())
    }

    /// Compute an arithmetic expression like "$error.line - 5" or "$error.line + 5"
    pub fn compute_expression(&self, expr: &str) -> Option<i64> {
        let expr = expr.trim();
        // Find $variable in expression
        for (prefix, op) in [(" + ", "add"), (" - ", "sub")] {
            if let Some(pos) = expr.find(prefix) {
                let left = expr[..pos].trim();
                let right = expr[pos + prefix.len()..].trim();
                let left_val = if left.starts_with('$') {
                    self.resolve_variable(left)?.as_i64()?
                } else {
                    left.parse::<i64>().ok()?
                };
                let right_val = if right.starts_with('$') {
                    self.resolve_variable(right)?.as_i64()?
                } else {
                    right.parse::<i64>().ok()?
                };
                return match op {
                    "add" => Some(left_val + right_val),
                    "sub" => Some(left_val - right_val),
                    _ => None,
                };
            }
        }
        // No operator found — try as a direct variable reference
        if expr.starts_with('$') {
            return self.resolve_variable(expr)?.as_i64();
        }
        // Try as a literal number
        expr.parse::<i64>().ok()
    }
}

// ============================================================================
// Messages
// ============================================================================

/// Messages for the ToolOrchestrator actor.
#[derive(Debug)]
pub enum ToolOrchestratorMessage {
    /// Execute a single tool by step name.
    ExecuteTool {
        tool_name: String,
        parameters: HashMap<String, String>,
        reply_to: tokio::sync::oneshot::Sender<Result<String>>,
    },
    /// Execute an ordered chain of tools sequentially.
    ExecuteToolChain {
        tools: Vec<String>,
        parameters: HashMap<String, String>,
        reply_to: tokio::sync::oneshot::Sender<Result<Vec<String>>>,
    },
    /// Execute a tool chain with full build error context for variable resolution.
    ExecuteToolChainWithContext {
        tools: Vec<String>,
        error: BuildError,
        project_root: String,
        reply_to: tokio::sync::oneshot::Sender<Result<Vec<String>>>,
    },
}

// ============================================================================
// ToolOrchestrator
// ============================================================================

/// The ToolOrchestrator actor — executes tools via graph-driven step definitions.
pub struct ToolOrchestrator {
    /// Sender to the MemoryGraphActor for all graph queries.
    memory_graph_tx: mpsc::Sender<MemoryGraphMessage>,
    /// Sender to the TransportActor (for VSC extension tools).
    transport_tx: mpsc::Sender<TransportMessage>,
    /// Sender to the MCP client actor (for MCP server tools).
    mcp_tx: mpsc::Sender<McpClientMessage>,
    /// Sender to the LLM actor (for analysis steps).
    llm_tx: mpsc::Sender<LlmMessage>,
    /// Sender to the ToolRouter actor (for project meta-tools).
    tool_router_tx: mpsc::Sender<crate::actors::tool_providers::ToolRouterMessage>,
}

impl ToolOrchestrator {
    pub fn new(
        memory_graph_tx: mpsc::Sender<MemoryGraphMessage>,
        transport_tx: mpsc::Sender<TransportMessage>,
        mcp_tx: mpsc::Sender<McpClientMessage>,
        llm_tx: mpsc::Sender<LlmMessage>,
        tool_router_tx: mpsc::Sender<crate::actors::tool_providers::ToolRouterMessage>,
    ) -> Self {
        Self {
            memory_graph_tx,
            transport_tx,
            mcp_tx,
            llm_tx,
            tool_router_tx,
        }
    }

    // ── Step Execution ─────────────────────────────────────────────────

    /// Execute a single tool step by looking up the StepDefinition from the graph.
    async fn execute_step(&self, step_name: &str, context: &StepContext) -> Result<String> {
        info!("ToolOrchestrator: executing step: {}", step_name);

        // 1. Query the StepDefinition node from the graph
        let step_def = match self.query_step_definition(step_name).await {
            Ok(def) => def,
            Err(_) => {
                // StepDefinition not registered in the graph (e.g. an
                // LLM-generated plan step). Fall back to dispatching the step
                // name through the generic ToolRouter so the underlying tool
                // (filesystem_write, etc.) is still executed, and the plan
                // approval doesn't fail outright.
                warn!(
                    "ToolOrchestrator: StepDefinition '{}' not found in graph — falling back to generic tool dispatch",
                    step_name
                );
                // Try generic ToolRouter dispatch; if the tool isn't
                // registered anywhere, return an error rather than silently
                // recording the step as completed.
                use crate::actors::tool_providers::ToolRouterMessage;
                let (tx, rx) = tokio::sync::oneshot::channel();
                if self
                    .tool_router_tx
                    .send(ToolRouterMessage::CallTool {
                        tool_name: step_name.to_string(),
                        args: serde_json::json!({}),
                        reply_to: tx,
                    })
                    .await
                    .is_ok()
                {
                    if let Ok(Ok(result)) = rx.await {
                        return Ok(serde_json::to_string(&result)
                            .unwrap_or_else(|_| result.to_string()));
                    }
                }
                warn!(
                    "ToolOrchestrator: no provider for unregistered step '{}' — returning error instead of a silent stub",
                    step_name
                );
                return Err(anyhow::anyhow!(
                    "no provider registered for step '{}' (tool not found)",
                    step_name
                ));
            }
        };

        // 2. Query the ToolProvider node for transport routing
        let provider_name = step_def
            .get("provider")
            .and_then(|v| v.as_str())
            .unwrap_or("vscode-extension");
        let concrete_tool = step_def
            .get("concrete_tool")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let arg_template = step_def
            .get("arg_template")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        // 3. Resolve the arg_template by substituting variables
        let resolved_args = self.resolve_template(&arg_template, context)?;

        // 4. Get the output_key for this step
        let output_key = step_def
            .get("output_key")
            .and_then(|v| v.as_str())
            .unwrap_or("result");

        // 5. Dispatch based on provider
        let result = match provider_name {
            "vscode-extension" => {
                self.dispatch_vscode_extension(concrete_tool, &resolved_args)
                    .await?
            }
            "mcp-cargo" | "mcp-node" | "mcp-filesystem" | "mcp-search" | "mcp-terminal" => {
                self.dispatch_mcp(provider_name, concrete_tool, &resolved_args)
                    .await?
            }
            "llm" => self.dispatch_llm(&resolved_args).await?,
            "project-meta" => {
                self.dispatch_project_meta(concrete_tool, &resolved_args)
                    .await?
            }
            _ => {
                warn!(
                    "ToolOrchestrator: unknown provider '{}' for step '{}', falling back to stub",
                    provider_name, step_name
                );
                format!(
                    "Step '{}' executed (stub for provider '{}')",
                    step_name, provider_name
                )
            }
        };

        info!(
            "ToolOrchestrator: step '{}' completed with output_key '{}'",
            step_name, output_key
        );
        Ok(result)
    }

    // ── Provider Dispatchers ───────────────────────────────────────────

    /// Dispatch to a VS Code extension tool via TransportActor.
    async fn dispatch_vscode_extension(
        &self,
        tool: &str,
        args: &serde_json::Value,
    ) -> Result<String> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.transport_tx
            .send(TransportMessage::CallExtension {
                method: tool.to_string(),
                params: args.clone(),
                reply_to: tx,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Transport send failed: {}", e))?;

        match rx.await {
            Ok(val) => Ok(serde_json::to_string(&val).unwrap_or_else(|_| format!("{:?}", val))),
            Err(e) => Err(anyhow::anyhow!("Transport response error: {}", e)),
        }
    }

    /// Dispatch to an MCP server tool via McpClientActor.
    async fn dispatch_mcp(
        &self,
        server_name: &str,
        tool: &str,
        args: &serde_json::Value,
    ) -> Result<String> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.mcp_tx
            .send(McpClientMessage::CallTool {
                server_name: server_name.to_string(),
                tool_name: tool.to_string(),
                arguments: args.as_object().cloned(),
                reply_to: tx,
            })
            .await
            .map_err(|e| anyhow::anyhow!("MCP send failed: {}", e))?;

        let result = rx
            .await
            .map_err(|e| anyhow::anyhow!("MCP response error: {}", e))?
            .map_err(|e| anyhow::anyhow!("MCP call failed: {}", e))?;

        let text = result
            .content
            .first()
            .and_then(|c| {
                // MCP content blocks are defined in rust_mcp_sdk::schema::ContentBlock
                use rust_mcp_sdk::schema::ContentBlock;
                match c {
                    ContentBlock::TextContent(text_content) => Some(text_content.text.clone()),
                    _ => None,
                }
            })
            .unwrap_or_else(|| "{}".to_string());

        Ok(text)
    }

    /// Dispatch to the LLM for analysis steps.
    async fn dispatch_llm(&self, args: &serde_json::Value) -> Result<String> {
        let system = args
            .get("system")
            .and_then(|v| v.as_str())
            .unwrap_or("You are a helpful assistant.");
        let user = args.get("user").and_then(|v| v.as_str()).unwrap_or("");

        let (tx, rx) = tokio::sync::oneshot::channel();
        self.llm_tx
            .send(LlmMessage::Complete {
                prompt: format!("{}\n\nUser: {}", system, user),
                role: crate::subsystems::llm::llm::LlmModelRole::Default,
                reply_to: tx,
            })
            .await
            .map_err(|e| anyhow::anyhow!("LLM send failed: {}", e))?;

        let response = rx
            .await
            .map_err(|e| anyhow::anyhow!("LLM response error: {}", e))?
            .map_err(|e| anyhow::anyhow!("LLM completion failed: {}", e))?;

        Ok(response)
    }

    /// Dispatch to a project meta-tool (project/build, project/test, etc.) via ToolRouter.
    async fn dispatch_project_meta(&self, tool: &str, args: &serde_json::Value) -> Result<String> {
        use crate::actors::tool_providers::ToolRouterMessage;

        let (tx, rx) = tokio::sync::oneshot::channel();
        self.tool_router_tx
            .send(ToolRouterMessage::CallTool {
                tool_name: format!("project/{}", tool),
                args: args.clone(),
                reply_to: tx,
            })
            .await
            .map_err(|e| anyhow::anyhow!("ToolRouter send failed: {}", e))?;

        let result = rx
            .await
            .map_err(|e| anyhow::anyhow!("ToolRouter response error: {}", e))?
            .map_err(|e| anyhow::anyhow!("ToolRouter call failed: {}", e))?;

        Ok(serde_json::to_string(&result).unwrap_or_else(|_| result.to_string()))
    }

    // ── Template Engine ────────────────────────────────────────────────

    /// Resolve a template value by substituting $variable references.
    /// Recursively walks the JSON value tree.
    fn resolve_template(
        &self,
        template: &serde_json::Value,
        context: &StepContext,
    ) -> Result<serde_json::Value> {
        match template {
            serde_json::Value::String(s) => {
                // Check if the entire string is a $variable expression
                if s.starts_with('$') {
                    // Try arithmetic expression first
                    if s.contains(" + ") || s.contains(" - ") {
                        if let Some(val) = context.compute_expression(s) {
                            return Ok(serde_json::json!(val));
                        }
                    }
                    // Try as a direct variable reference
                    if let Some(val) = context.resolve_variable(s) {
                        return Ok(val);
                    }
                    // Fall back to treating it as a literal string with potential inline vars
                    let resolved = self.resolve_inline_variables(s, context);
                    return Ok(serde_json::Value::String(resolved));
                }
                // Resolve any inline $variable references
                let resolved = self.resolve_inline_variables(s, context);
                Ok(serde_json::Value::String(resolved))
            }
            serde_json::Value::Object(map) => {
                let mut new_map = serde_json::Map::new();
                for (k, v) in map {
                    new_map.insert(k.clone(), self.resolve_template(v, context)?);
                }
                Ok(serde_json::Value::Object(new_map))
            }
            serde_json::Value::Array(arr) => {
                let mut new_arr = Vec::new();
                for v in arr {
                    new_arr.push(self.resolve_template(v, context)?);
                }
                Ok(serde_json::Value::Array(new_arr))
            }
            other => Ok(other.clone()),
        }
    }

    /// Resolve $variable references inline within a string (e.g. "Error: $error.error_text").
    fn resolve_inline_variables(&self, s: &str, context: &StepContext) -> String {
        let mut result = String::new();
        let mut chars = s.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '$' {
                // Collect the variable name after $
                let mut var_name = String::new();
                while let Some(&next) = chars.peek() {
                    if next.is_alphanumeric()
                        || next == '.'
                        || next == '['
                        || next == ']'
                        || next == '_'
                    {
                        var_name.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
                if !var_name.is_empty() {
                    let full_var = format!("${}", var_name);
                    if let Some(val) = context.resolve_variable(&full_var) {
                        result.push_str(&val.to_string());
                    } else {
                        result.push_str(&full_var);
                    }
                } else {
                    result.push('$');
                }
            } else {
                result.push(c);
            }
        }

        result
    }

    // ── Graph Queries ──────────────────────────────────────────────────

    /// Query a StepDefinition node from the graph by step name.
    async fn query_step_definition(
        &self,
        name: &str,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.memory_graph_tx
            .send(MemoryGraphMessage::QueryAttrNodes {
                node_type: Some("stepDefinition".to_string()),
                subtype: None,
                name: Some(name.to_string()),
                limit: Some(1),
                reply_to: tx,
            })
            .await?;

        let nodes = rx.await??;
        if let Some(attr) = nodes.into_iter().next() {
            Ok(attr.properties.clone())
        } else {
            Err(anyhow::anyhow!(
                "StepDefinition '{}' not found in graph",
                name
            ))
        }
    }

    // ── Tool Chain Execution ───────────────────────────────────────────

    /// Execute an ordered chain of tools sequentially, with context propagation.
    async fn execute_tool_chain_with_context(
        &self,
        tools: &[String],
        error: &BuildError,
        project_root: &str,
    ) -> Result<Vec<String>> {
        info!(
            "ToolOrchestrator: executing tool chain with {} tools (graph-driven)",
            tools.len()
        );

        let mut context = StepContext::new(error, project_root);
        let mut results = Vec::new();

        for step_name in tools {
            // Look up the step definition to check dependencies
            let step_def = self.query_step_definition(step_name).await?;

            // Check dependencies
            if let Some(deps) = step_def.get("depends_on").and_then(|v| v.as_array()) {
                for dep in deps {
                    if let Some(dep_name) = dep.as_str() {
                        if !context.step_outputs.contains_key(dep_name) {
                            warn!(
                                "ToolOrchestrator: dependency '{}' not satisfied for step '{}'",
                                dep_name, step_name
                            );
                            // Non-fatal: continue anyway with best-effort
                        }
                    }
                }
            }

            // Execute the step
            let result = self.execute_step(step_name, &context).await?;

            // Store the output if the step has an output_key
            if let Some(output_key) = step_def.get("output_key").and_then(|v| v.as_str()) {
                // Try to parse as JSON for structured output
                let value: serde_json::Value = serde_json::from_str(&result)
                    .unwrap_or(serde_json::Value::String(result.clone()));
                context.set_output(output_key, value);
            }

            results.push(result);
        }

        Ok(results)
    }

    // ── Legacy methods (for backward compatibility) ─────────────────────

    /// Execute a single tool (legacy / plan-step path).
    /// Plan steps carry concrete parameters (e.g. path, content for
    /// filesystem_write) that must be forwarded to the ToolRouter. The graph
    /// StepDefinition lookup is skipped because plan steps reference real
    /// routable tools (filesystem_*, build_*) that are not registered as
    /// StepDefinition nodes.
    async fn execute_tool(
        &self,
        tool_name: &str,
        parameters: &HashMap<String, String>,
    ) -> Result<String> {
        // Convert the step parameters into a JSON object for ToolRouter args.
        use crate::actors::tool_providers::ToolRouterMessage;
        let args: serde_json::Value = parameters
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect::<serde_json::Map<_, _>>()
            .into();

        let (tx, rx) = tokio::sync::oneshot::channel();
        if self
            .tool_router_tx
            .send(ToolRouterMessage::CallTool {
                tool_name: tool_name.to_string(),
                args,
                reply_to: tx,
            })
            .await
            .is_ok()
        {
            if let Ok(Ok(result)) = rx.await {
                return Ok(serde_json::to_string(&result)
                    .unwrap_or_else(|_| result.to_string()));
            }
        }

        // Fallback: routing failed (no StepDefinition, no registered provider).
        // Return an ERROR so callers can distinguish a genuinely handled tool
        // from an unknown one — silently swallowing unknowns hides real
        // misconfigurations and broke plan-step validation tests.
        Err(anyhow::anyhow!(
            "No provider registered for tool '{}' — the tool could not be executed",
            tool_name
        ))
    }

    /// Execute an ordered chain of tools (legacy path — minimal context).
    async fn execute_tool_chain(
        &self,
        tools: &[String],
        _parameters: &HashMap<String, String>,
    ) -> Result<Vec<String>> {
        info!(
            "ToolOrchestrator: executing tool chain with {} tools (legacy)",
            tools.len()
        );

        let default_context = StepContext::default();
        let mut results = Vec::new();

        for step_name in tools {
            let result = self.execute_step(step_name, &default_context).await?;
            results.push(result);
        }

        Ok(results)
    }
}

#[async_trait]
impl Actor for ToolOrchestrator {
    type Message = ToolOrchestratorMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            ToolOrchestratorMessage::ExecuteTool {
                tool_name,
                parameters,
                reply_to,
            } => {
                let result = self.execute_tool(&tool_name, &parameters).await;
                let _ = reply_to.send(result);
            }
            ToolOrchestratorMessage::ExecuteToolChain {
                tools,
                parameters,
                reply_to,
            } => {
                let result = self.execute_tool_chain(&tools, &parameters).await;
                let _ = reply_to.send(result);
            }
            ToolOrchestratorMessage::ExecuteToolChainWithContext {
                tools,
                error,
                project_root,
                reply_to,
            } => {
                let result = self
                    .execute_tool_chain_with_context(&tools, &error, &project_root)
                    .await;
                let _ = reply_to.send(result);
            }
        }
    }
}

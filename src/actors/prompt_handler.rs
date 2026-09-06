// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! PromptHandlerActor — handles the LLM prompt lifecycle with context injection.
//!
//! This actor replaces the inline prompt handling in CoordinatorActor with a
//! dedicated actor that:
//!
//! 1. Receives a user query + optional context (intent, build state, etc.)
//! 2. Gathers relevant context from the memory graph (project context, memories,
//!    related intents, error types, fix strategies)
//! 3. Builds a system prompt with all gathered context
//! 4. Sends the prompt to the LLM actor (with or without tools)
//! 5. If tools are provided, handles the tool call lifecycle (parse, execute, follow-up)
//! 6. Returns the LLM response
//!
//! This keeps the prompt lifecycle isolated and testable, and ensures the
//! CoordinatorActor stays focused on orchestration.

use async_trait::async_trait;
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot};
use tracing::info;

use crate::actors::system_prompt::SystemPromptMessage;
use crate::actors::tool_providers::ToolRouterMessage;
use crate::actors::Actor;
use crate::models::memory_graph::SearchOptions;
use crate::subsystems::graph::memory_graph::MemoryGraphMessage;
use crate::subsystems::llm::llm::LlmMessage;

// ============================================================================
// PromptContext — context injected into the prompt
// ============================================================================

/// Context gathered for a prompt.
#[derive(Debug, Clone, Default)]
pub struct PromptContext {
    /// The matched intent name, if any.
    pub intent_name: Option<String>,
    /// Project context from the memory graph.
    pub project_context: Option<String>,
    /// Relevant memories from the memory graph.
    pub memories: Vec<String>,
    /// Related error types (for build/error intents).
    pub error_types: Vec<String>,
    /// Related fix strategies (for build/error intents).
    pub fix_strategies: Vec<String>,
    /// Available tools relevant to the intent.
    pub relevant_tools: Vec<String>,
    /// Build state information.
    pub build_state: Option<String>,
}

// ============================================================================
// Messages
// ============================================================================

/// Messages for the PromptHandlerActor.
#[derive(Debug)]
pub enum PromptHandlerMessage {
    /// Handle a user query by building a prompt with context and sending to LLM.
    HandlePrompt {
        /// The user's query text.
        query: String,
        /// Optional matched intent name (from IntentRouterActor).
        intent_name: Option<String>,
        /// Optional build context (from BuildOrchestrator).
        build_context: Option<HashMap<String, String>>,
        /// Reply channel for the LLM response.
        reply_to: oneshot::Sender<Result<String, String>>,
    },
    /// Handle a user query with tool support (full tool call lifecycle).
    HandlePromptWithTools {
        /// The user's query text.
        query: String,
        /// Optional matched intent name (from IntentRouterActor).
        intent_name: Option<String>,
        /// Optional build context (from BuildOrchestrator).
        build_context: Option<HashMap<String, String>>,
        /// Available tools for the LLM to use.
        tools: Vec<crate::actors::ToolInfo>,
        /// Reply channel for the LLM response.
        reply_to: oneshot::Sender<Result<String, String>>,
    },
}

// ============================================================================
// Actor
// ============================================================================

/// The PromptHandlerActor — handles LLM prompt lifecycle with context injection.
pub struct PromptHandlerActor {
    memory_graph_tx: mpsc::Sender<MemoryGraphMessage>,
    llm_tx: mpsc::Sender<LlmMessage>,
    system_prompt_tx: mpsc::Sender<SystemPromptMessage>,
    tool_router_tx: mpsc::Sender<ToolRouterMessage>,
}

impl PromptHandlerActor {
    pub fn new(
        memory_graph_tx: mpsc::Sender<MemoryGraphMessage>,
        llm_tx: mpsc::Sender<LlmMessage>,
        system_prompt_tx: mpsc::Sender<SystemPromptMessage>,
        tool_router_tx: mpsc::Sender<ToolRouterMessage>,
    ) -> Self {
        Self {
            memory_graph_tx,
            llm_tx,
            system_prompt_tx,
            tool_router_tx,
        }
    }

    /// Handle a user query: gather context, build prompt, send to LLM.
    async fn handle_prompt(
        &self,
        query: String,
        intent_name: Option<String>,
        build_context: Option<HashMap<String, String>>,
    ) -> Result<String, String> {
        info!(
            "[PROMPT_HANDLER] handling prompt (intent: {:?})",
            intent_name
        );

        // Step 1: Gather context from the memory graph
        info!("[PROMPT_HANDLER] STEP 1: gathering context from memory graph");
        let context = self
            .gather_context(&query, &intent_name, build_context)
            .await;

        // Step 2: Build the system prompt with context
        info!("[PROMPT_HANDLER] STEP 2: building system prompt");
        let system_prompt = self.build_system_prompt(&context).await?;
        info!(
            "[PROMPT_HANDLER] System prompt built ({} chars)",
            system_prompt.len()
        );

        // Step 3: Build the full prompt (system + user query)
        let full_prompt = format!("{}\n\n## User Query\n{}", system_prompt, query);
        info!(
            "[PROMPT_HANDLER] STEP 3: full prompt ready ({} chars)",
            full_prompt.len()
        );

        // Step 4: Send to LLM
        info!("[PROMPT_HANDLER] STEP 4: sending to LLM");
        let response = self.send_to_llm(&full_prompt).await?;

        info!(
            "[PROMPT_HANDLER] LLM response received ({} chars)",
            response.len()
        );
        Ok(response)
    }

    /// Handle a user query with tool support: gather context, build prompt,
    /// send to LLM with tools, execute tool calls, follow up for final response.
    async fn handle_prompt_with_tools(
        &self,
        query: String,
        intent_name: Option<String>,
        build_context: Option<HashMap<String, String>>,
        tools: Vec<crate::actors::ToolInfo>,
    ) -> Result<String, String> {
        info!(
            "[PROMPT_HANDLER] handling prompt with tools (intent: {:?}, {} tools)",
            intent_name,
            tools.len()
        );

        // Step 1: Gather context from the memory graph
        info!("[PROMPT_HANDLER] STEP 1: gathering context from memory graph");
        let context = self
            .gather_context(&query, &intent_name, build_context)
            .await;

        // Step 2: Build the system prompt with context
        info!("[PROMPT_HANDLER] STEP 2: building system prompt with context");
        let system_prompt = self.build_system_prompt(&context).await?;
        info!(
            "[PROMPT_HANDLER] System prompt built ({} chars)",
            system_prompt.len()
        );

        // Step 3: Build the full prompt (system + user query)
        let full_prompt = format!("{}\n\n## User Query\n{}", system_prompt, query);
        info!(
            "[PROMPT_HANDLER] STEP 3: full prompt ready ({} chars)",
            full_prompt.len()
        );

        // Step 4: Build messages array (system prompt as a system message, then user query)
        let messages = vec![crate::subsystems::chat::chat::ChatMessageData {
            id: "system-prompt".to_string(),
            role: "system".to_string(),
            content: full_prompt,
            timestamp: chrono::Utc::now().to_rfc3339(),
            widget: None,
        }];

        // Step 5: Send to LLM with tools
        info!(
            "[PROMPT_HANDLER] STEP 5: sending to LLM with {} tools",
            tools.len()
        );
        let (tx, rx) = oneshot::channel();
        self.llm_tx
            .send(LlmMessage::CompleteWithTools {
                messages,
                tools,
                reply_to: tx,
            })
            .await
            .map_err(|e| format!("Failed to send to LLM: {}", e))?;

        let llm_response = rx
            .await
            .map_err(|e| format!("LLM oneshot error: {}", e))?
            .map_err(|e| format!("LLM error: {}", e))?;

        info!(
            "[PROMPT_HANDLER] LLM first response received ({} chars)",
            llm_response.len()
        );

        // Step 6: Check if the response contains tool_calls (JSON format)
        let final_content = if let Ok(json_msg) =
            serde_json::from_str::<serde_json::Value>(&llm_response)
        {
            let has_tc = json_msg
                .get("tool_calls")
                .and_then(|t| t.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            info!(
                "[PROMPT_HANDLER] STEP 6: JSON response, has {} tool_calls",
                has_tc
            );
            if let Some(tool_calls) = json_msg["tool_calls"].as_array() {
                if !tool_calls.is_empty() {
                    // Execute each tool call and collect results
                    let mut tool_results: Vec<serde_json::Value> = Vec::new();

                    for tc in tool_calls {
                        let function_name = tc["function"]["name"].as_str().unwrap_or("unknown");
                        let function_args: serde_json::Value = tc["function"]["arguments"]
                            .as_str()
                            .and_then(|s| serde_json::from_str(s).ok())
                            .unwrap_or(serde_json::Value::Null);
                        let tool_call_id = tc["id"].as_str().unwrap_or("call_unknown");

                        info!(
                            "[PROMPT_HANDLER] EXECUTING tool: {} (id={})",
                            function_name, tool_call_id
                        );

                        let (tool_result, duration_ms) = self
                            .execute_tool_call(function_name, function_args, tool_call_id)
                            .await;

                        match &tool_result {
                            Ok(result) => {
                                info!(
                                    "[PROMPT_HANDLER] TOOL {}: success ({}ms)",
                                    function_name, duration_ms
                                );
                                tool_results.push(serde_json::json!({
                                    "tool_call_id": tool_call_id,
                                    "tool_name": function_name,
                                    "result": result,
                                }));
                            }
                            Err(e) => {
                                info!(
                                    "[PROMPT_HANDLER] TOOL {}: FAILED ({}ms): {}",
                                    function_name, duration_ms, e
                                );
                                tool_results.push(serde_json::json!({
                                    "tool_call_id": tool_call_id,
                                    "tool_name": function_name,
                                    "error": e,
                                }));
                            }
                        }
                    }

                    // Append tool results as a new user message and get final response
                    let tool_results_text = serde_json::to_string_pretty(&tool_results)
                        .unwrap_or_else(|_| "[]".to_string());

                    let follow_up_messages = vec![
                        crate::subsystems::chat::chat::ChatMessageData {
                            id: "system-prompt".to_string(),
                            role: "system".to_string(),
                            content: system_prompt,
                            timestamp: chrono::Utc::now().to_rfc3339(),
                            widget: None,
                        },
                        crate::subsystems::chat::chat::ChatMessageData {
                            id: "user-query".to_string(),
                            role: "user".to_string(),
                            content: query.clone(),
                            timestamp: chrono::Utc::now().to_rfc3339(),
                            widget: None,
                        },
                        crate::subsystems::chat::chat::ChatMessageData {
                            id: "tool-results".to_string(),
                            role: "user".to_string(),
                            content: format!("Tool execution results:\n{}", tool_results_text),
                            timestamp: chrono::Utc::now().to_rfc3339(),
                            widget: None,
                        },
                    ];

                    // Send follow-up request to LLM with tool results
                    info!(
                        "[PROMPT_HANDLER] FOLLOW-UP: sending {} messages with tool results to LLM",
                        follow_up_messages.len()
                    );
                    let (tx2, rx2) = oneshot::channel();
                    self.llm_tx
                        .send(LlmMessage::CompleteWithMessages {
                            messages: follow_up_messages,
                            reply_to: tx2,
                        })
                        .await
                        .map_err(|e| format!("Failed to send follow-up to LLM: {}", e))?;

                    rx2.await
                        .map_err(|e| format!("LLM follow-up oneshot error: {}", e))?
                        .map_err(|e| format!("LLM follow-up error: {}", e))?
                } else {
                    // No tool calls — extract content
                    let content = json_msg["content"]
                        .as_str()
                        .unwrap_or(&llm_response)
                        .to_string();
                    info!(
                        "[PROMPT_HANDLER] No tool calls, extracting content ({} chars)",
                        content.len()
                    );
                    content
                }
            } else {
                // JSON but no tool_calls — check for content field
                let content = json_msg["content"]
                    .as_str()
                    .unwrap_or(&llm_response)
                    .to_string();
                info!(
                    "[PROMPT_HANDLER] JSON but no tool_calls, extracting content field ({} chars)",
                    content.len()
                );
                content
            }
        } else {
            // Plain text response — no tool calls
            info!(
                "[PROMPT_HANDLER] Plain text LLM response ({} chars)",
                llm_response.len()
            );
            llm_response
        };

        info!(
            "[PROMPT_HANDLER] final response ready ({} chars)",
            final_content.len()
        );
        Ok(final_content)
    }

    /// Execute a tool call by routing through the ToolRouterActor.
    async fn execute_tool_call(
        &self,
        tool_name: &str,
        args: serde_json::Value,
        _tool_call_id: &str,
    ) -> (Result<serde_json::Value, String>, u64) {
        let start = std::time::Instant::now();

        // Route through the ToolRouterActor
        let (tx, rx) = tokio::sync::oneshot::channel();
        let result = if self
            .tool_router_tx
            .send(ToolRouterMessage::CallTool {
                tool_name: tool_name.to_string(),
                args: args.clone(),
                reply_to: tx,
            })
            .await
            .is_ok()
        {
            match rx.await {
                Ok(result) => result,
                Err(e) => Err(format!("ToolRouter response error: {}", e)),
            }
        } else {
            Err("ToolRouter actor not available".to_string())
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        (result, duration_ms)
    }

    /// Gather context from the memory graph for the given query and intent.
    async fn gather_context(
        &self,
        query: &str,
        intent_name: &Option<String>,
        _build_context: Option<HashMap<String, String>>,
    ) -> PromptContext {
        let mut context = PromptContext::default();
        context.intent_name = intent_name.clone();

        // Gather project context
        if let Some(project_ctx) = self.get_project_context().await {
            info!(
                "[PROMPT_HANDLER] Project context: {} chars",
                project_ctx.len()
            );
            context.project_context = Some(project_ctx);
        } else {
            info!("[PROMPT_HANDLER] No project context available");
        }

        // Gather relevant memories
        context.memories = self.search_memories(query).await;
        info!(
            "[PROMPT_HANDLER] Memories: {} items found",
            context.memories.len()
        );

        // If we have a specific intent, gather related information
        if let Some(intent) = intent_name {
            info!(
                "[PROMPT_HANDLER] Gathering related info for intent='{}'",
                intent
            );
            // Gather error types related to this intent
            context.error_types = self.query_related_nodes(intent, "build_error").await;
            info!(
                "[PROMPT_HANDLER] Error types: {} found",
                context.error_types.len()
            );

            // Gather fix strategies related to this intent
            context.fix_strategies = self.query_related_nodes(intent, "fix_strategy").await;
            info!(
                "[PROMPT_HANDLER] Fix strategies: {} found",
                context.fix_strategies.len()
            );

            // Gather relevant tools
            context.relevant_tools = self.query_related_nodes(intent, "tool").await;
            info!(
                "[PROMPT_HANDLER] Relevant tools: {} found",
                context.relevant_tools.len()
            );

            // Gather build state
            context.build_state = self.get_build_state(intent).await;
            info!("[PROMPT_HANDLER] Build state: {:?}", context.build_state);
        } else {
            info!("[PROMPT_HANDLER] No intent specified, skipping intent-specific context");
        }

        info!("[PROMPT_HANDLER] Context gathered: proj_ctx={}, memories={}, error_types={}, fix_strategies={}, tools={}, build_state={}",
            context.project_context.as_ref().map(|s| s.len()).unwrap_or(0),
            context.memories.len(),
            context.error_types.len(),
            context.fix_strategies.len(),
            context.relevant_tools.len(),
            context.build_state.is_some());

        context
    }

    /// Build the system prompt with all gathered context.
    async fn build_system_prompt(&self, context: &PromptContext) -> Result<String, String> {
        // Use BuildPrefix with empty tools to get the base system prompt
        let (tx, rx) = oneshot::channel();
        self.system_prompt_tx
            .send(SystemPromptMessage::BuildPrefix {
                tools_hash: 0,
                tools: Vec::new(),
                reply_to: tx,
            })
            .await
            .map_err(|e| format!("Failed to send BuildPrefix: {}", e))?;

        let prefix_messages = rx
            .await
            .map_err(|e| format!("BuildPrefix oneshot error: {}", e))?;
        let base_prompt = prefix_messages
            .into_iter()
            .map(|m| m.content)
            .collect::<Vec<_>>()
            .join("\n\n");

        // Build context sections
        let mut context_sections = Vec::new();

        if let Some(intent) = &context.intent_name {
            context_sections.push(format!("## Matched Intent\n{}", intent));
        }

        if let Some(project_ctx) = &context.project_context {
            context_sections.push(format!("## Project Context\n{}", project_ctx));
        }

        if !context.memories.is_empty() {
            context_sections.push(format!(
                "## Relevant Memories\n{}",
                context.memories.join("\n")
            ));
        }

        if !context.error_types.is_empty() {
            context_sections.push(format!(
                "## Related Error Types\n{}",
                context.error_types.join("\n")
            ));
        }

        if !context.fix_strategies.is_empty() {
            context_sections.push(format!(
                "## Available Fix Strategies\n{}",
                context.fix_strategies.join("\n")
            ));
        }

        if !context.relevant_tools.is_empty() {
            context_sections.push(format!(
                "## Relevant Tools\n{}",
                context.relevant_tools.join("\n")
            ));
        }

        if let Some(state) = &context.build_state {
            context_sections.push(format!("## Build State\n{}", state));
        }

        // Combine base prompt with context sections
        if context_sections.is_empty() {
            Ok(base_prompt)
        } else {
            Ok(format!(
                "{}\n\n## Context\n{}",
                base_prompt,
                context_sections.join("\n\n")
            ))
        }
    }

    /// Send a prompt to the LLM actor and get the response.
    async fn send_to_llm(&self, prompt: &str) -> Result<String, String> {
        let (tx, rx) = oneshot::channel();
        self.llm_tx
            .send(LlmMessage::Complete {
                prompt: prompt.to_string(),
                role: crate::subsystems::llm::llm::LlmModelRole::Default,
                reply_to: tx,
            })
            .await
            .map_err(|e| format!("Failed to send to LLM: {}", e))?;

        rx.await
            .map_err(|e| format!("LLM oneshot error: {}", e))?
            .map_err(|e| format!("LLM error: {}", e))
    }

    // ── Graph query helpers ─────────────────────────────────

    /// Get project context from the memory graph.
    async fn get_project_context(&self) -> Option<String> {
        let (tx, rx) = oneshot::channel();
        if self
            .memory_graph_tx
            .send(MemoryGraphMessage::GetProjectContext { reply_to: tx })
            .await
            .is_err()
        {
            return None;
        }
        match rx.await {
            Ok(Ok(snapshot)) => Some(format!("{:?}", snapshot)),
            _ => None,
        }
    }

    /// Search memories relevant to the query.
    async fn search_memories(&self, query: &str) -> Vec<String> {
        let (tx, rx) = oneshot::channel();
        if self
            .memory_graph_tx
            .send(MemoryGraphMessage::SearchContext {
                query: query.to_string(),
                options: Some(SearchOptions {
                    top_k: Some(5),
                    threshold: Some(0.3),
                    node_types: None,
                    max_depth: None,
                    include_structural: Some(true),
                    recency_weight: None,
                }),
                reply_to: tx,
            })
            .await
            .is_err()
        {
            return Vec::new();
        }
        match rx.await {
            Ok(Ok(result)) => result
                .nodes
                .into_iter()
                .map(|n| format!("{} (score: {:.2})", n.node.name(), n.score))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Query nodes related to a given intent name by subtype.
    async fn query_related_nodes(&self, _intent_name: &str, subtype: &str) -> Vec<String> {
        let (tx, rx) = oneshot::channel();
        if self
            .memory_graph_tx
            .send(MemoryGraphMessage::QueryAttrNodes {
                node_type: Some("Standard".to_string()),
                subtype: Some(subtype.to_string()),
                name: None,
                limit: Some(10),
                reply_to: tx,
            })
            .await
            .is_err()
        {
            return Vec::new();
        }
        match rx.await {
            Ok(Ok(nodes)) => nodes
                .into_iter()
                .map(|n| format!("{}: {}", n.name(), n.description().unwrap_or("")))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Get build state information.
    async fn get_build_state(&self, _intent_name: &str) -> Option<String> {
        let (tx, rx) = oneshot::channel();
        if self
            .memory_graph_tx
            .send(MemoryGraphMessage::QueryAttrNodes {
                node_type: Some("Standard".to_string()),
                subtype: Some("build_state".to_string()),
                name: None,
                limit: Some(1),
                reply_to: tx,
            })
            .await
            .is_err()
        {
            return None;
        }
        match rx.await {
            Ok(Ok(nodes)) => nodes
                .first()
                .map(|n| format!("{}: {}", n.name(), n.description().unwrap_or(""))),
            _ => None,
        }
    }
}

#[async_trait]
impl Actor for PromptHandlerActor {
    type Message = PromptHandlerMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            PromptHandlerMessage::HandlePrompt {
                query,
                intent_name,
                build_context,
                reply_to,
            } => {
                let result = self.handle_prompt(query, intent_name, build_context).await;
                let _ = reply_to.send(result);
            }
            PromptHandlerMessage::HandlePromptWithTools {
                query,
                intent_name,
                build_context,
                tools,
                reply_to,
            } => {
                let result = self
                    .handle_prompt_with_tools(query, intent_name, build_context, tools)
                    .await;
                let _ = reply_to.send(result);
            }
        }
    }
}

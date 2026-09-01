// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! ToolRouterActor — generic tool dispatch backed by a [`ToolRegistry`].
//!
//! Routing is fully generic: `ListTools` returns the registered tools plus the
//! dynamic MCP servers; `CallTool` dispatches through the registry's registered
//! handlers (built by the app composer — `spire_code::actors::tool_providers::build_default_registry`)
//! for unknown names. The registry starts empty.

pub mod registry;

pub use registry::{ToolHandler, ToolRegistry};

use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

use crate::subsystems::mcp::mcp_client::McpClientMessage;
use crate::actors::ToolInfo;

/// Messages for the ToolRouterActor.
pub enum ToolRouterMessage {
    /// List all registered tools plus dynamic MCP server tools.
    ListTools {
        reply_to: oneshot::Sender<Vec<ToolInfo>>,
    },
    /// Call a tool by exact name, routing through the registry (or MCP).
    CallTool {
        tool_name: String,
        args: Value,
        reply_to: oneshot::Sender<Result<Value, String>>,
    },
}

/// Generic tool dispatcher.
///
/// Owns no tool definitions — every tool is registered into the shared
/// [`ToolRegistry`] by the app composer (generic tools from this crate, coding
/// tools from `spire-code`). MCP server tools are the one dynamic backend and
/// are queried per request.
pub struct ToolRouterActor {
    /// Registered tools (populated by the composer at startup).
    registry: Arc<ToolRegistry>,
    /// Sender to the McpClientActor (dynamic MCP servers + catch-all routing).
    mcp_client_tx: mpsc::Sender<McpClientMessage>,
}

impl ToolRouterActor {
    /// Create a router backed by the given (pre-populated) registry.
    pub fn new(
        registry: Arc<ToolRegistry>,
        mcp_client_tx: mpsc::Sender<McpClientMessage>,
    ) -> Self {
        Self {
            registry,
            mcp_client_tx,
        }
    }

    /// List tools from the MCP client (connected servers).
    /// Filters out the pseudo "spire" server since its tools are already
    /// covered by the registered set. MCP tool names are prefixed with their
    /// server name (e.g. "mcp-cargo/build").
    async fn list_mcp_tools(&self) -> Vec<ToolInfo> {
        let (tx, rx) = oneshot::channel();
        if self
            .mcp_client_tx
            .send(McpClientMessage::GetConnectedServersWithTools { reply_to: tx })
            .await
            .is_err()
        {
            return Vec::new();
        }

        match rx.await {
            Ok(servers) => {
                let mut tools = Vec::new();
                for (server_name, server_tools) in servers {
                    if server_name == "spire" {
                        continue;
                    }
                    for t in server_tools {
                        tools.push(ToolInfo {
                            name: format!("{}/{}", server_name, t.name),
                            description: t.description.unwrap_or_default(),
                            input_schema: serde_json::to_value(t.input_schema)
                                .unwrap_or(serde_json::Value::Null),
                        });
                    }
                }
                tools
            }
            Err(_) => Vec::new(),
        }
    }

    /// Call an MCP tool via the McpClientActor.
    ///
    /// MCP tool names are expected to be prefixed with their server name
    /// (e.g. "mcp-cargo/build"). This method parses the prefix to route the
    /// call to the correct MCP server.
    async fn call_mcp_tool(&self, tool_name: &str, args: Value) -> Result<Value, String> {
        let (server_name, actual_tool_name) = match tool_name.split_once('/') {
            Some((server, tool)) => (server.to_string(), tool.to_string()),
            None => (String::new(), tool_name.to_string()),
        };

        let (tx, rx) = oneshot::channel();
        self.mcp_client_tx
            .send(McpClientMessage::CallTool {
                server_name,
                tool_name: actual_tool_name,
                arguments: args.as_object().cloned(),
                reply_to: tx,
            })
            .await
            .map_err(|e| format!("MCP client send error: {}", e))?;

        match rx.await {
            Ok(Ok(result)) => serde_json::to_value(result)
                .map_err(|e| format!("MCP result serialization error: {}", e)),
            Ok(Err(e)) => Err(format!("MCP tool call error: {}", e)),
            Err(_) => Err("MCP client response error".to_string()),
        }
    }
}

#[async_trait]
impl crate::actors::Actor for ToolRouterActor {
    type Message = ToolRouterMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            ToolRouterMessage::ListTools { reply_to } => {
                let mut all_tools = self.registry.list();
                all_tools.extend(self.list_mcp_tools().await);
                let _ = reply_to.send(all_tools);
            }

            ToolRouterMessage::CallTool {
                tool_name,
                args,
                reply_to,
            } => {
                let result = match self.registry.call(&tool_name, args.clone()) {
                    Some(fut) => fut.await,
                    None => self.call_mcp_tool(&tool_name, args).await,
                };
                let _ = reply_to.send(result);
            }
        }
    }
}

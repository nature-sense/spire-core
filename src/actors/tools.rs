// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! ToolsActor — manages tool registry and dispatch.
//!
//! This actor stores registered tools and handles tool invocation requests.
//! Tools can be registered from external MCP servers or from the VS Code extension.
//!
//! Routing is handled by the `ToolRouterActor`, which holds senders to all
//! tool backends (extension, embedded, MCP) and routes tool calls by prefix
//! matching. This replaces the previous `ToolDispatcher` + `ToolProvider` pattern.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use tokio::sync::mpsc;

use crate::actors::tool_providers::ToolRouterMessage;
use crate::actors::{Actor, ActorError, ToolInfo};

/// A registered tool with its handler.
pub struct RegisteredTool {
    pub info: ToolInfo,
    /// The server that provides this tool.
    pub server: String,
}

/// Messages for the Tools actor.
pub enum ToolsMessage {
    /// Register a tool.
    RegisterTool {
        server: String,
        info: ToolInfo,
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
    /// Unregister all tools from a server.
    UnregisterServer {
        server: String,
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
    /// List all registered tools.
    ListTools {
        reply_to: tokio::sync::oneshot::Sender<Vec<ToolInfo>>,
    },
    /// Call a tool by name.
    CallTool {
        tool: String,
        args: Value,
        reply_to: tokio::sync::oneshot::Sender<Result<Value, ActorError>>,
    },
    /// Register all VS Code extension tools at once.
    RegisterVscodeTools {
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
}

/// Actor that manages tool registration and dispatch.
pub struct ToolsActor {
    tools: HashMap<String, RegisteredTool>,
    /// Sender to the ToolRouterActor for routing tool calls.
    tool_router_tx: mpsc::Sender<ToolRouterMessage>,
}

impl ToolsActor {
    pub fn new(tool_router_tx: mpsc::Sender<ToolRouterMessage>) -> Self {
        Self {
            tools: HashMap::new(),
            tool_router_tx,
        }
    }
}

#[async_trait]
impl Actor for ToolsActor {
    type Message = ToolsMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            ToolsMessage::RegisterTool {
                server,
                info,
                reply_to,
            } => {
                self.tools
                    .insert(info.name.clone(), RegisteredTool { info, server });
                let _ = reply_to.send(Ok(()));
            }
            ToolsMessage::UnregisterServer { server, reply_to } => {
                self.tools.retain(|_, t| t.server != server);
                let _ = reply_to.send(Ok(()));
            }
            ToolsMessage::ListTools { reply_to } => {
                // Use the ToolRouterActor to get all tools from all backends.
                let (tx, rx) = tokio::sync::oneshot::channel();
                if self
                    .tool_router_tx
                    .send(ToolRouterMessage::ListTools { reply_to: tx })
                    .await
                    .is_ok()
                {
                    let mut tools = rx.await.unwrap_or_default();
                    // Merge tools registered directly with this actor
                    // (RegisterTool) so they stay visible alongside the
                    // router's built-in/MCP tools.
                    for t in self.tools.values() {
                        if !tools.iter().any(|existing| existing.name == t.info.name) {
                            tools.push(t.info.clone());
                        }
                    }
                    let _ = reply_to.send(tools);
                } else {
                    let _ = reply_to.send(vec![]);
                }
            }
            ToolsMessage::CallTool {
                tool,
                args,
                reply_to,
            } => {
                let result = self.call_tool(&tool, args).await;
                let _ = reply_to.send(result);
            }
            ToolsMessage::RegisterVscodeTools { reply_to } => {
                // VS Code tools are already registered via the ExtensionToolProvider
                // in the ToolRouterActor. This message is kept for backward compatibility.
                let _ = reply_to.send(Ok(()));
            }
        }
    }
}

impl ToolsActor {
    async fn call_tool(&self, tool_name: &str, args: Value) -> Result<Value, ActorError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self
            .tool_router_tx
            .send(ToolRouterMessage::CallTool {
                tool_name: tool_name.to_string(),
                args,
                reply_to: tx,
            })
            .await
            .is_err()
        {
            return Err(ActorError::Internal(
                "ToolRouter actor not available".to_string(),
            ));
        }
        match rx.await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(e)) => Err(ActorError::Internal(e)),
            Err(_) => Err(ActorError::Internal(
                "ToolRouter response error".to_string(),
            )),
        }
    }
}

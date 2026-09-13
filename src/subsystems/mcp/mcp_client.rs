// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! McpClientActor — wraps external MCP server connections behind message-passing.
//!
//! This actor owns the `McpClientManager` and processes connection/disconnection
//! and tool-call requests asynchronously. It also maintains a set of internal
//! tools that appear under the pseudo-MCP server name "spire".

use async_trait::async_trait;
use rust_mcp_sdk::schema::{CallToolResult, Tool};
use serde::Serialize;
use std::time::Instant;

use crate::actors::progress::{ProgressMessage, ProgressStatus, ProgressUpdate};
use crate::actors::{Actor, ActorError};
use crate::mcp::client::{
    BuildSystemInfo, McpClientManager, McpServerConfig, CONNECT_TIMEOUT_SECS,
};

/// Structured detail about an MCP server for the UI.
#[derive(Debug, Clone, Serialize)]
pub struct McpServerDetail {
    pub name: String,
    pub description: String,
    pub server_type: String,
    pub tool_count: usize,
    pub properties: serde_json::Value,
    /// Build system this server serves (e.g. "Cargo", "npm"). Absent for
    /// general-purpose servers. Lets the UI group servers under build systems.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub build_type: Option<String>,
}

/// Messages for the MCP Client actor.
pub enum McpClientMessage {
    /// Load config from a list of server entries (from the graph database).
    /// This replaces the file-based config loading with graph-stored configs.
    LoadConfigFromGraph {
        servers: Vec<McpServerConfig>,
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
    /// Add a single server configuration programmatically.
    AddConfig {
        config: McpServerConfig,
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
    /// Connect to all configured servers.
    ConnectAll {
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
    /// Connect to a single server by name.
    Connect {
        server_name: String,
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
    /// Disconnect from all servers.
    DisconnectAll {
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
    /// Disconnect from a single server.
    Disconnect {
        server_name: String,
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
    /// Get the list of tools exposed by a connected server.
    /// For the pseudo-server "spire", returns internal tools.
    GetTools {
        server_name: String,
        reply_to: tokio::sync::oneshot::Sender<Option<Vec<Tool>>>,
    },
    /// Get the names of all connected servers.
    ConnectedServers {
        reply_to: tokio::sync::oneshot::Sender<Vec<String>>,
    },
    /// Get all connected servers with their discovered tools.
    GetConnectedServersWithTools {
        reply_to: tokio::sync::oneshot::Sender<Vec<(String, Vec<rust_mcp_sdk::schema::Tool>)>>,
    },
    /// Get structured details about all servers (for the UI).
    /// Includes the pseudo "spire" server with internal tools.
    GetServerDetails {
        reply_to: tokio::sync::oneshot::Sender<Vec<McpServerDetail>>,
    },
    /// Call a tool on a specific external MCP server.
    CallTool {
        server_name: String,
        tool_name: String,
        arguments: Option<serde_json::Map<String, serde_json::Value>>,
        reply_to: tokio::sync::oneshot::Sender<Result<CallToolResult, ActorError>>,
    },
    /// Set the internal tools that appear under the pseudo "spire" server.
    SetInternalTools {
        tools: Vec<Tool>,
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
    /// Get build system info for a connected server.
    GetBuildSystemInfo {
        server_name: String,
        reply_to: tokio::sync::oneshot::Sender<Option<BuildSystemInfo>>,
    },
    /// Get all connected servers that have build system info (build MCPs).
    GetBuildServers {
        reply_to: tokio::sync::oneshot::Sender<Vec<(String, BuildSystemInfo)>>,
    },
}

/// Actor that wraps `McpClientManager` behind message-passing.
///
/// Also holds a set of internal tools that are presented as a pseudo-MCP
/// server named "spire" in the UI.
pub struct McpClientActor {
    manager: McpClientManager,
    /// Internal tools exposed under the pseudo "spire" server.
    internal_tools: Vec<Tool>,
    /// Optional sender for publishing progress notifications on tool calls.
    progress_tx: Option<tokio::sync::mpsc::Sender<ProgressMessage>>,
}

impl McpClientActor {
    pub fn new() -> Self {
        Self {
            manager: McpClientManager::new(),
            internal_tools: Vec::new(),
            progress_tx: None,
        }
    }

    /// Create a new McpClientActor with a progress notification channel.
    /// When set, tool calls will publish start/completion notifications
    /// via the ProgressActor broadcast channel.
    pub fn with_progress(progress_tx: tokio::sync::mpsc::Sender<ProgressMessage>) -> Self {
        Self {
            manager: McpClientManager::new(),
            internal_tools: Vec::new(),
            progress_tx: Some(progress_tx),
        }
    }
}

impl Default for McpClientActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for McpClientActor {
    type Message = McpClientMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            McpClientMessage::LoadConfigFromGraph { servers, reply_to } => {
                self.manager.load_config_from_entries(servers);
                let _ = reply_to.send(Ok(()));
            }
            McpClientMessage::AddConfig { config, reply_to } => {
                self.manager.add_config(config);
                let _ = reply_to.send(Ok(()));
            }
            McpClientMessage::ConnectAll { reply_to } => {
                self.manager.connect_all().await;
                let _ = reply_to.send(Ok(()));
            }
            McpClientMessage::Connect {
                server_name,
                reply_to,
            } => {
                // Bounded on purpose: this actor's mailbox is shared by every
                // tool call, so a board that is powered off (or firewalled) must
                // not park it. Same budget as `connect_all`.
                let result = match tokio::time::timeout(
                    std::time::Duration::from_secs(CONNECT_TIMEOUT_SECS),
                    self.manager.connect(&server_name),
                )
                .await
                {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(e)) => Err(ActorError::Internal(format!("Failed to connect: {}", e))),
                    Err(_) => Err(ActorError::Internal(format!(
                        "Timed out connecting to '{}' after {}s",
                        server_name, CONNECT_TIMEOUT_SECS
                    ))),
                };
                let _ = reply_to.send(result);
            }
            McpClientMessage::DisconnectAll { reply_to } => {
                self.manager.disconnect_all().await;
                let _ = reply_to.send(Ok(()));
            }
            McpClientMessage::Disconnect {
                server_name,
                reply_to,
            } => {
                self.manager.disconnect(&server_name).await;
                let _ = reply_to.send(Ok(()));
            }
            McpClientMessage::GetTools {
                server_name,
                reply_to,
            } => {
                // Check the pseudo "spire" server first
                if server_name == "spire" {
                    let _ = reply_to.send(Some(self.internal_tools.clone()));
                    return;
                }
                let tools = self.manager.get_tools(&server_name).map(|t| t.to_vec());
                let _ = reply_to.send(tools);
            }
            McpClientMessage::ConnectedServers { reply_to } => {
                let mut servers: Vec<String> = self
                    .manager
                    .connected_servers()
                    .into_iter()
                    .map(|s| s.to_string())
                    .collect();
                // Always include the pseudo "spire" server if it has tools
                if !self.internal_tools.is_empty() {
                    servers.push("spire".to_string());
                }
                let _ = reply_to.send(servers);
            }
            McpClientMessage::GetConnectedServersWithTools { reply_to } => {
                let mut result: Vec<(String, Vec<rust_mcp_sdk::schema::Tool>)> = self
                    .manager
                    .connected_servers()
                    .into_iter()
                    .map(|name| {
                        let tools = self
                            .manager
                            .get_tools(name)
                            .map(|t| t.to_vec())
                            .unwrap_or_default();
                        (name.to_string(), tools)
                    })
                    .collect();
                // Always include the pseudo "spire" server if it has tools
                if !self.internal_tools.is_empty() {
                    result.push(("spire".to_string(), self.internal_tools.clone()));
                }
                let _ = reply_to.send(result);
            }
            McpClientMessage::GetServerDetails { reply_to } => {
                let details = self.build_server_details();
                let _ = reply_to.send(details);
            }
            McpClientMessage::CallTool {
                server_name,
                tool_name,
                arguments,
                reply_to,
            } => {
                let task_id = format!("mcp:{}:{}", server_name, tool_name);

                // Publish "running" notification before the call
                if let Some(ref tx) = self.progress_tx {
                    let _ = tx
                        .send(ProgressMessage::Publish {
                            update: ProgressUpdate {
                                task_id: task_id.clone(),
                                message: format!("Calling {} on {}", tool_name, server_name),
                                percent: 0.0,
                                status: ProgressStatus::Running,
                                metadata: Some(serde_json::json!({
                                    "server": server_name,
                                    "tool": tool_name,
                                })),
                            },
                        })
                        .await;
                }

                let start = Instant::now();
                let result = self
                    .manager
                    .call_tool(&server_name, &tool_name, arguments)
                    .await
                    .map_err(|e| ActorError::Internal(format!("Tool call failed: {}", e)));
                let elapsed = start.elapsed();

                // Publish "completed" or "failed" notification after the call
                if let Some(ref tx) = self.progress_tx {
                    let (status, message) = match &result {
                        Ok(_) => (
                            ProgressStatus::Completed,
                            format!(
                                "{} on {} completed in {:.1}s",
                                tool_name,
                                server_name,
                                elapsed.as_secs_f64()
                            ),
                        ),
                        Err(e) => (
                            ProgressStatus::Failed,
                            format!(
                                "{} on {} failed after {:.1}s: {}",
                                tool_name,
                                server_name,
                                elapsed.as_secs_f64(),
                                e
                            ),
                        ),
                    };
                    let _ = tx
                        .send(ProgressMessage::Publish {
                            update: ProgressUpdate {
                                task_id: task_id.clone(),
                                message,
                                percent: 100.0,
                                status,
                                metadata: Some(serde_json::json!({
                                    "server": server_name,
                                    "tool": tool_name,
                                    "elapsed_secs": elapsed.as_secs_f64(),
                                })),
                            },
                        })
                        .await;
                }

                let _ = reply_to.send(result);
            }
            McpClientMessage::SetInternalTools { tools, reply_to } => {
                self.internal_tools = tools;
                let _ = reply_to.send(Ok(()));
            }
            McpClientMessage::GetBuildSystemInfo {
                server_name,
                reply_to,
            } => {
                let info = self.manager.get_build_system_info(&server_name).cloned();
                let _ = reply_to.send(info);
            }
            McpClientMessage::GetBuildServers { reply_to } => {
                let servers: Vec<(String, BuildSystemInfo)> = self
                    .manager
                    .build_servers()
                    .into_iter()
                    .map(|(name, info)| (name.to_string(), info.clone()))
                    .collect();
                let _ = reply_to.send(servers);
            }
        }
    }
}

impl McpClientActor {
    /// Build structured server details for the UI, matching the format
    /// expected by the webview (see mock-env-server.mjs).
    ///
    /// Includes the pseudo "spire" server with internal tools.
    /// Shows ALL configured servers from the database — both online (successfully
    /// connected) and offline (failed to connect or not yet connected).
    fn build_server_details(&self) -> Vec<McpServerDetail> {
        // Collect all configured server names (from configs, which includes
        // everything loaded from the database, even if connection failed).
        let configured: Vec<&str> = self.manager.configured_servers();

        tracing::info!(
            "build_server_details: {} configured servers, {} connections, {} internal tools",
            configured.len(),
            self.manager.connected_servers().len(),
            self.internal_tools.len(),
        );

        let mut details: Vec<McpServerDetail> = configured
            .into_iter()
            .map(|name| {
                let is_online = self.manager.is_connected(name);
                let tools = if is_online {
                    self.manager
                        .get_tools(name)
                        .map(|t| t.to_vec())
                        .unwrap_or_default()
                } else {
                    vec![]
                };
                let tool_count = tools.len();

                // Include build system info in properties if available
                let build_info = self.manager.get_build_system_info(name).cloned();
                let mut props = serde_json::json!({
                    "status": if is_online { "online" } else { "offline" },
                });
                if let Some(ref bi) = build_info {
                    if let Some(ref analyzer) = bi.analyzer_tool {
                        props["analyzer_tool"] = serde_json::json!(analyzer);
                    }
                    props["build_systems"] = serde_json::json!(bi.build_systems);
                    props["capabilities"] = serde_json::json!(bi.capabilities);
                }

                tracing::info!(
                    "build_server_details: server '{}' status={} tools={} build={}",
                    name,
                    if is_online { "online" } else { "offline" },
                    tool_count,
                    build_info.is_some(),
                );

                McpServerDetail {
                    name: name.to_string(),
                    description: String::new(),
                    server_type: "external".to_string(),
                    tool_count,
                    properties: props,
                    build_type: self.manager.get_build_type(name).map(|s| s.to_string()),
                }
            })
            .collect();

        // Add the pseudo "spire" server with internal tools
        if !self.internal_tools.is_empty() {
            tracing::info!(
                "build_server_details: adding pseudo 'spire' server with {} internal tools",
                self.internal_tools.len(),
            );
            details.push(McpServerDetail {
                name: "spire".to_string(),
                description: "Built-in Spire tools (VS Code extension API)".to_string(),
                server_type: "internal".to_string(),
                tool_count: self.internal_tools.len(),
                properties: serde_json::json!({
                    "status": "online",
                }),
                build_type: None,
            });
        }

        details
    }
}

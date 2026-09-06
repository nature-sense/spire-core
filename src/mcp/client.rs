// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! MCP client for connecting to external MCP servers.
//!
//! This module provides `McpClientManager` which manages connections to
//! external MCP servers (filesystem, git, etc.) via stdio or HTTP transport.
//! It does NOT host an MCP server — it is purely a client.

use anyhow::Result;
use rust_mcp_sdk::mcp_client::{
    client_runtime_core::create_client, ClientHandlerCore, McpClientOptions, ToMcpClientHandlerCore,
};
use rust_mcp_sdk::schema::{
    CallToolRequestParams, CallToolResult, InitializeRequestParams, InitializeResult, Tool,
    LATEST_PROTOCOL_VERSION,
};
use rust_mcp_sdk::ClientStreamableTransport;
use rust_mcp_sdk::McpClient;
use rust_mcp_sdk::RequestOptions;
use rust_mcp_sdk::StdioTransport;
use rust_mcp_sdk::StreamableTransportOptions;
use rust_mcp_sdk::TransportOptions;
use std::collections::HashMap;

use rust_mcp_sdk::schema::RpcError;
use std::sync::Arc;
use tracing::{error, info, warn};

// ---------------------------------------------------------------------------
// BuildSystemInfo — self-description from MCP servers that can build projects
// ---------------------------------------------------------------------------

/// Metadata returned by a build MCP server's `_build_system` tool.
/// Describes what build systems this server supports and what it can do.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BuildSystemInfo {
    /// Server identifier (e.g. "mcp-cargo")
    pub name: String,
    /// Build system names this server handles (e.g. ["Cargo"])
    pub build_systems: Vec<String>,
    /// High-level project type (e.g. "rust", "node", "python")
    pub build_type: String,
    /// Standard tool capabilities (e.g. "build", "test", "clean", "check")
    pub capabilities: Vec<String>,
    /// Name of the analyze tool, if any
    pub analyzer_tool: Option<String>,
    /// Config file patterns this server can parse
    pub config_files: Vec<String>,
}

/// Generic transport configuration for connecting to an MCP server.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type")]
pub enum TransportConfig {
    /// Spawn a subprocess and communicate via stdin/stdout.
    #[serde(rename = "stdio")]
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
    },
    /// Connect to an existing HTTP(S) MCP server (Streamable HTTP).
    #[serde(rename = "http")]
    Http {
        url: String,
        #[serde(default)]
        headers: HashMap<String, String>,
    },
}

/// Configuration for an external MCP server connection.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub transport: TransportConfig,
    /// If false, the server is not started automatically.
    #[serde(default = "default_autostart")]
    pub autostart: bool,
    /// Build system this server serves (e.g. "Cargo", "npm").
    /// Used to scope tools to project build systems.
    #[serde(default)]
    pub build_type: Option<String>,
}

fn default_autostart() -> bool {
    true
}

/// Represents a live connection to an external MCP server.
struct McpClientConnection {
    config: McpServerConfig,
    runtime: Arc<dyn McpClient>,
    server_info: InitializeResult,
    tools: Vec<Tool>,
    /// Build system metadata, if the server exposes a `_build_system` tool.
    build_system: Option<BuildSystemInfo>,
}

/// Manages connections to external MCP servers.
///
/// This client connects to specialized MCP servers (e.g., filesystem, github, postgres)
/// and provides a unified interface for discovering their capabilities and calling tools on them.
pub struct McpClientManager {
    /// All configured server configs (including those not yet connected).
    configs: HashMap<String, McpServerConfig>,
    /// Live connections to servers that have successfully connected.
    connections: HashMap<String, McpClientConnection>,
}

impl McpClientManager {
    /// Create a new, empty manager.
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            connections: HashMap::new(),
        }
    }

    /// Load config from a list of server entries (from the graph database).
    /// This replaces the file-based config loading with graph-stored configs.
    /// Clears any existing configs and replaces them with the provided entries.
    pub fn load_config_from_entries(&mut self, servers: Vec<McpServerConfig>) {
        // Clear existing configs and connections
        self.configs.clear();
        self.connections.clear();

        for server_config in servers {
            let transport_desc = match &server_config.transport {
                TransportConfig::Stdio { command, args, .. } => {
                    format!("{} {}", command, args.join(" "))
                }
                TransportConfig::Http { url, .. } => {
                    format!("http {}", url)
                }
            };
            info!(
                "MCP Client: loaded config for server '{}' ({})",
                server_config.name, transport_desc,
            );

            // Track in configs
            self.configs
                .insert(server_config.name.clone(), server_config.clone());

            self.connections
                .entry(server_config.name.clone())
                .or_insert_with(|| McpClientConnection {
                    config: server_config,
                    runtime: Arc::new(PlaceholderClient),
                    server_info: InitializeResult {
                        server_info: rust_mcp_sdk::schema::Implementation {
                            name: String::new(),
                            version: String::new(),
                            description: None,
                            icons: vec![],
                            website_url: None,
                            title: None,
                        },
                        capabilities: rust_mcp_sdk::schema::ServerCapabilities::default(),
                        protocol_version: String::new(),
                        instructions: None,
                        meta: None,
                    },
                    tools: vec![],
                    build_system: None,
                });
        }
    }

    /// Add or replace a server configuration programmatically.
    pub fn add_config(&mut self, config: McpServerConfig) {
        self.configs.insert(config.name.clone(), config.clone());
        self.connections.insert(
            config.name.clone(),
            McpClientConnection {
                config,
                runtime: Arc::new(PlaceholderClient),
                server_info: InitializeResult {
                    server_info: rust_mcp_sdk::schema::Implementation {
                        name: String::new(),
                        version: String::new(),
                        description: None,
                        icons: vec![],
                        website_url: None,
                        title: None,
                    },
                    capabilities: rust_mcp_sdk::schema::ServerCapabilities::default(),
                    protocol_version: String::new(),
                    instructions: None,
                    meta: None,
                },
                tools: vec![],
                build_system: None,
            },
        );
    }

    /// Connect to all configured servers.
    ///
    /// Each server connection has a 15-second timeout. If a server subprocess
    /// hangs during initialization, it will be skipped so other servers can
    /// still connect.
    pub async fn connect_all(&mut self) {
        let names: Vec<String> = self.connections.keys().cloned().collect();
        for name in names {
            if let Some(conn) = self.connections.get(&name) {
                if !conn.config.autostart {
                    info!(
                        "MCP Client: skipping '{}' (autostart=false, externally managed)",
                        name
                    );
                    continue;
                }
            }
            let name_clone = name.clone();
            match tokio::time::timeout(
                std::time::Duration::from_secs(15),
                self.connect(&name_clone),
            )
            .await
            {
                Ok(Ok(())) => {
                    info!("MCP Client: connected to '{}'", name_clone);
                }
                Ok(Err(e)) => {
                    error!("MCP Client: failed to connect to '{}': {:#}", name_clone, e);
                }
                Err(_) => {
                    warn!(
                        "MCP Client: timed out connecting to '{}' after 15s, skipping",
                        name_clone
                    );
                }
            }
        }
    }

    /// Connect to a single server by name.
    pub async fn connect(&mut self, server_name: &str) -> Result<()> {
        let config = self
            .connections
            .get(server_name)
            .map(|c| c.config.clone())
            .ok_or_else(|| anyhow::anyhow!("Unknown MCP server: {}", server_name))?;

        let client_details = InitializeRequestParams {
            protocol_version: LATEST_PROTOCOL_VERSION.into(),
            capabilities: rust_mcp_sdk::schema::ClientCapabilities::default(),
            client_info: rust_mcp_sdk::schema::Implementation {
                name: "spire-core".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                description: None,
                icons: vec![],
                website_url: None,
                title: None,
            },
            meta: None,
        };

        let handler = SimpleClientHandler;

        let runtime: Arc<dyn McpClient> = match &config.transport {
            TransportConfig::Stdio { command, args, env } => {
                info!(
                    "MCP Client: connecting to '{}' via stdio ({} {})",
                    server_name,
                    command,
                    args.join(" ")
                );

                // Inject SPIRE_LOG_DIR into the subprocess environment so that
                // MCP servers can write their own log files.
                let mut merged_env = env.clone();
                if let Ok(log_dir) = std::env::var("SPIRE_LOG_DIR") {
                    merged_env.insert("SPIRE_LOG_DIR".to_string(), log_dir);
                }

                let transport = StdioTransport::create_with_server_launch(
                    command.clone(),
                    args.clone(),
                    Some(merged_env),
                    TransportOptions::default(),
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to create stdio transport for '{}': {}",
                        server_name,
                        e
                    )
                })?;

                create_client(McpClientOptions {
                    client_details,
                    transport,
                    handler: handler.to_mcp_client_handler(),
                    task_store: None,
                    server_task_store: None,
                    message_observer: None,
                })
            }
            TransportConfig::Http { url, headers } => {
                info!(
                    "MCP Client: connecting to '{}' via HTTP ({})",
                    server_name, url
                );

                let transport_options = StreamableTransportOptions {
                    mcp_url: url.clone(),
                    request_options: RequestOptions {
                        custom_headers: if headers.is_empty() {
                            None
                        } else {
                            Some(headers.clone())
                        },
                        ..Default::default()
                    },
                };

                let transport = ClientStreamableTransport::new(
                    &transport_options,
                    None, // session_id
                    true, // standalone
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to create HTTP transport for '{}': {}",
                        server_name,
                        e
                    )
                })?;

                create_client(McpClientOptions {
                    client_details,
                    transport,
                    handler: handler.to_mcp_client_handler(),
                    task_store: None,
                    server_task_store: None,
                    message_observer: None,
                })
            }
        };

        // Apply a 15-second timeout to the start() call to prevent hanging
        // if the subprocess doesn't respond during initialization.
        tokio::time::timeout(std::time::Duration::from_secs(15), runtime.clone().start())
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "Timed out starting MCP client for '{}' after 15s",
                    server_name
                )
            })?
            .map_err(|e| {
                anyhow::anyhow!("Failed to start MCP client for '{}': {}", server_name, e)
            })?;

        let server_info = runtime.server_info().ok_or_else(|| {
            anyhow::anyhow!("Server '{}' did not return server info", server_name)
        })?;

        info!(
            "MCP Client: connected to '{}' (v{}, tools: {:?})",
            server_info.server_info.name,
            server_info.server_info.version,
            server_info.capabilities.tools.is_some()
        );

        let tools = if server_info.capabilities.tools.is_some() {
            match tokio::time::timeout(
                std::time::Duration::from_secs(10),
                runtime.request_tool_list(None),
            )
            .await
            {
                Ok(Ok(list)) => {
                    info!(
                        "MCP Client: '{}' exposes {} tools",
                        server_name,
                        list.tools.len()
                    );
                    list.tools
                }
                Ok(Err(e)) => {
                    warn!(
                        "MCP Client: failed to list tools for '{}': {}",
                        server_name, e
                    );
                    vec![]
                }
                Err(_) => {
                    warn!(
                        "MCP Client: timed out listing tools for '{}' after 10s",
                        server_name
                    );
                    vec![]
                }
            }
        } else {
            vec![]
        };

        // Discover build system metadata via the `_build_system` tool convention.
        // If the server exposes this tool, call it once and store the result.
        let build_system = if tools.iter().any(|t| t.name == "_build_system") {
            info!(
                "MCP Client: '{}' exposes _build_system tool, querying...",
                server_name
            );
            match runtime
                .request_tool_call(CallToolRequestParams {
                    name: "_build_system".into(),
                    arguments: None,
                    meta: None,
                    task: None,
                })
                .await
            {
                Ok(result) => {
                    // Parse the first text content as JSON
                    let text = result
                        .content
                        .iter()
                        .filter_map(|c| {
                            if let rust_mcp_sdk::schema::ContentBlock::TextContent(tc) = c {
                                Some(tc.text.clone())
                            } else {
                                None
                            }
                        })
                        .next()
                        .unwrap_or_default();
                    match serde_json::from_str::<BuildSystemInfo>(&text) {
                        Ok(info) => {
                            info!(
                                "MCP Client: '{}' build system info: {:?}",
                                server_name, info.build_systems
                            );
                            Some(info)
                        }
                        Err(e) => {
                            warn!(
                                "MCP Client: failed to parse _build_system response from '{}': {}",
                                server_name, e
                            );
                            None
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "MCP Client: failed to call _build_system on '{}': {}",
                        server_name, e
                    );
                    None
                }
            }
        } else {
            None
        };

        self.connections.insert(
            server_name.to_string(),
            McpClientConnection {
                config,
                runtime,
                server_info,
                tools,
                build_system,
            },
        );

        Ok(())
    }

    /// Disconnect from all servers.
    pub async fn disconnect_all(&mut self) {
        let names: Vec<String> = self.connections.keys().cloned().collect();
        for name in names {
            self.disconnect(&name).await;
        }
    }

    /// Disconnect from a single server.
    pub async fn disconnect(&mut self, server_name: &str) {
        if let Some(conn) = self.connections.get(server_name) {
            info!("MCP Client: disconnecting from '{}'", server_name);
            if let Err(e) = conn.runtime.shut_down().await {
                warn!("MCP Client: error shutting down '{}': {}", server_name, e);
            }
        }
        self.connections.remove(server_name);
    }

    /// Get the server info for a connected server.
    pub fn get_server_info(&self, server_name: &str) -> Option<&InitializeResult> {
        self.connections.get(server_name).map(|c| &c.server_info)
    }

    /// Get the list of tools exposed by a connected server.
    pub fn get_tools(&self, server_name: &str) -> Option<&[Tool]> {
        self.connections
            .get(server_name)
            .map(|c| c.tools.as_slice())
    }

    /// Get the build system info for a connected server, if it exposed a `_build_system` tool.
    pub fn get_build_system_info(&self, server_name: &str) -> Option<&BuildSystemInfo> {
        self.connections
            .get(server_name)
            .and_then(|c| c.build_system.as_ref())
    }

    /// Get all connected servers that have build system info (i.e. are build MCPs).
    pub fn build_servers(&self) -> Vec<(&str, &BuildSystemInfo)> {
        self.connections
            .iter()
            .filter_map(|(name, conn)| conn.build_system.as_ref().map(|bs| (name.as_str(), bs)))
            .collect()
    }

    /// Get the names of all connected servers.
    pub fn connected_servers(&self) -> Vec<&str> {
        self.connections.keys().map(|s| s.as_str()).collect()
    }

    /// Get the names of all configured servers (including those not yet connected).
    pub fn configured_servers(&self) -> Vec<&str> {
        self.configs.keys().map(|s| s.as_str()).collect()
    }

    /// The build type a configured server serves, if any (e.g. "Cargo", "npm").
    /// Returns None for general-purpose servers.
    pub fn get_build_type(&self, name: &str) -> Option<&str> {
        self.configs.get(name).and_then(|c| c.build_type.as_deref())
    }

    /// Check whether a server has a real (non-placeholder) connection.
    /// Returns false if the server is not in connections, or if it's still
    /// a placeholder (failed to connect / not yet connected).
    pub fn is_connected(&self, server_name: &str) -> bool {
        self.connections
            .get(server_name)
            .map(|c| !c.server_info.server_info.name.is_empty())
            .unwrap_or(false)
    }

    /// Call a tool on a specific external MCP server.
    ///
    /// If `server_name` is empty, searches all connected servers for a server
    /// that exposes a tool with the given `tool_name`.
    pub async fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: Option<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<CallToolResult> {
        // If server_name is empty, find the server that has this tool
        let resolved_server = if server_name.is_empty() {
            self.find_server_for_tool(tool_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "No connected MCP server found that provides tool '{}'",
                    tool_name
                )
            })?
        } else {
            server_name.to_string()
        };

        let conn = self.connections.get(&resolved_server).ok_or_else(|| {
            anyhow::anyhow!("Unknown or disconnected MCP server: {}", resolved_server)
        })?;

        info!(
            "MCP Client: calling '{}' on '{}'",
            tool_name, resolved_server
        );

        let result = conn
            .runtime
            .request_tool_call(CallToolRequestParams {
                name: tool_name.into(),
                arguments,
                meta: None,
                task: None,
            })
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "Tool call '{}' on '{}' failed: {}",
                    tool_name,
                    resolved_server,
                    e
                )
            })?;

        Ok(result)
    }

    /// Find a connected server that exposes a tool with the given name.
    fn find_server_for_tool(&self, tool_name: &str) -> Option<String> {
        for (name, conn) in &self.connections {
            if conn.tools.iter().any(|t| t.name == tool_name) {
                return Some(name.clone());
            }
        }
        None
    }
}

impl Default for McpClientManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Placeholder client used before real connections are established
// ---------------------------------------------------------------------------
struct PlaceholderClient;

#[async_trait::async_trait]
impl McpClient for PlaceholderClient {
    async fn start(self: Arc<Self>) -> rust_mcp_sdk::error::SdkResult<()> {
        Ok(())
    }
    fn set_server_details(&self, _: InitializeResult) -> rust_mcp_sdk::error::SdkResult<()> {
        Ok(())
    }
    async fn terminate_session(&self) {}
    fn task_store(&self) -> Option<Arc<rust_mcp_sdk::task_store::ClientTaskStore>> {
        None
    }
    fn server_task_store(&self) -> Option<Arc<rust_mcp_sdk::task_store::ServerTaskStore>> {
        None
    }
    async fn shut_down(&self) -> rust_mcp_sdk::error::SdkResult<()> {
        Ok(())
    }
    async fn is_shut_down(&self) -> bool {
        true
    }
    fn client_info(&self) -> &InitializeRequestParams {
        static FALLBACK: std::sync::LazyLock<InitializeRequestParams> =
            std::sync::LazyLock::new(|| InitializeRequestParams {
                protocol_version: LATEST_PROTOCOL_VERSION.into(),
                capabilities: rust_mcp_sdk::schema::ClientCapabilities::default(),
                client_info: rust_mcp_sdk::schema::Implementation {
                    name: "spire-core".into(),
                    version: "0.1.0".into(),
                    description: None,
                    icons: vec![],
                    website_url: None,
                    title: None,
                },
                meta: None,
            });
        &FALLBACK
    }
    fn server_info(&self) -> Option<InitializeResult> {
        None
    }
    async fn session_id(&self) -> Option<rust_mcp_sdk::SessionId> {
        None
    }
    async fn send(
        &self,
        _: rust_mcp_sdk::schema::schema_utils::MessageFromClient,
        _: Option<rust_mcp_sdk::schema::RequestId>,
        _: Option<std::time::Duration>,
    ) -> rust_mcp_sdk::error::SdkResult<Option<rust_mcp_sdk::schema::schema_utils::ServerMessage>>
    {
        Ok(None)
    }
    async fn send_batch(
        &self,
        _: Vec<rust_mcp_sdk::schema::schema_utils::ClientMessage>,
        _: Option<std::time::Duration>,
    ) -> rust_mcp_sdk::error::SdkResult<
        Option<Vec<rust_mcp_sdk::schema::schema_utils::ServerMessage>>,
    > {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// Simple client handler — does nothing for incoming requests/notifications
// ---------------------------------------------------------------------------
struct SimpleClientHandler;

#[async_trait::async_trait]
impl ClientHandlerCore for SimpleClientHandler {
    async fn handle_request(
        &self,
        _request: rust_mcp_sdk::schema::schema_utils::ServerJsonrpcRequest,
        _runtime: &dyn McpClient,
    ) -> std::result::Result<
        rust_mcp_sdk::schema::schema_utils::ResultFromClient,
        rust_mcp_sdk::schema::RpcError,
    > {
        Err(rust_mcp_sdk::schema::RpcError::internal_error()
            .with_message("No request handler implemented".to_string()))
    }

    async fn handle_notification(
        &self,
        _notification: rust_mcp_sdk::schema::schema_utils::NotificationFromServer,
        _runtime: &dyn McpClient,
    ) -> std::result::Result<(), rust_mcp_sdk::schema::RpcError> {
        Ok(())
    }

    async fn handle_error(
        &self,
        error: &rust_mcp_sdk::schema::RpcError,
        _runtime: &dyn McpClient,
    ) -> std::result::Result<(), rust_mcp_sdk::schema::RpcError> {
        // Surface the real server error for debugging, then treat it as handled
        // so the SDK doesn't fabricate a secondary error on top of it.
        warn!(
            "MCP server returned error (code={}, message={}) — continuing",
            error.code, error.message
        );
        Ok(())
    }

    /// Override the default `handle_process_error` to log at `warn!` level instead of `error!`.
    ///
    /// Some MCP servers (e.g., `@modelcontextprotocol/server-filesystem`) write informational
    /// messages to stdout (e.g., "Secure MCP Filesystem Server running on stdio") which the
    /// SDK's stdio transport treats as process errors. These are not actual errors — they are
    /// harmless banner/info messages that should not clutter the error log.
    async fn handle_process_error(
        &self,
        error_message: String,
        runtime: &dyn McpClient,
    ) -> std::result::Result<(), RpcError> {
        if !runtime.is_shut_down().await {
            warn!("Process message (non-fatal): {error_message}");
        }
        Ok(())
    }
}

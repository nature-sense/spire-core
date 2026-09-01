// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! MCP subsystem — MCP client manager (external tool servers).

pub mod mcp_client;
pub use mcp_client::{McpClientActor, McpClientMessage};

use tokio::sync::mpsc;

use crate::actors::progress::ProgressMessage;
use crate::actors::Actor;
use spire_actor::registry::ServiceRegistry;
use spire_actor::subsystem::Subsystem;

/// Handles for the MCP subsystem.
pub struct McpHandles {
    /// Sender to the MCP client manager actor.
    pub tx: mpsc::Sender<McpClientMessage>,
}

/// Cohesive actor group for MCP integration.
pub struct McpSubsystem {
    /// Optional progress reporter the MCP client reports into.
    pub progress_tx: Option<mpsc::Sender<ProgressMessage>>,
}

impl Subsystem for McpSubsystem {
    type Handles = McpHandles;

    fn spawn(self, registry: &ServiceRegistry) -> Self::Handles {
        let (tx, rx) = mpsc::channel(64);
        let _ = registry.register::<McpClientMessage>("mcp.client", tx.clone());
        // Compatibility alias.
        let _ = registry.register::<McpClientMessage>("mcp_client", tx.clone());
        let actor = match self.progress_tx {
            Some(progress_tx) => McpClientActor::with_progress(progress_tx),
            None => McpClientActor::new(),
        };
        let _join = actor.spawn(rx);
        McpHandles { tx }
    }

    fn actor_count(&self) -> usize {
        1
    }
}

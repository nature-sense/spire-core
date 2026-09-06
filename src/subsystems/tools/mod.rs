// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Tools subsystem — tool orchestration, providers, watcher.

pub mod file_watcher;
pub use file_watcher::{
    FileChangeKind, FileChangeNotification, FileEventBatch, FileEventInfo, FileWatcherActor,
    FileWatcherMessage,
};
pub mod tool_orchestrator;
pub use tool_orchestrator::{ToolOrchestrator, ToolOrchestratorMessage};

use tokio::sync::mpsc;

use crate::actors::tool_providers::ToolRouterMessage;
use crate::actors::Actor;
use crate::subsystems::graph::memory_graph::MemoryGraphMessage;
use crate::subsystems::llm::llm::LlmMessage;
use crate::subsystems::mcp::mcp_client::McpClientMessage;
use crate::transport::socket::TransportMessage;
use spire_actor::registry::ServiceRegistry;
use spire_actor::subsystem::Subsystem;

/// Handles for the tools subsystem.
pub struct ToolsHandles {
    /// Sender to the tool orchestrator actor.
    pub orchestrator_tx: mpsc::Sender<ToolOrchestratorMessage>,
    /// Sender to the file watcher actor.
    pub watcher_tx: mpsc::Sender<FileWatcherMessage>,
}

/// Dependencies the tools subsystem needs from other subsystems.
pub struct ToolsDeps {
    pub memory_graph_tx: mpsc::Sender<MemoryGraphMessage>,
    pub transport_tx: mpsc::Sender<TransportMessage>,
    pub mcp_tx: mpsc::Sender<McpClientMessage>,
    pub llm_tx: mpsc::Sender<LlmMessage>,
    pub tool_router_tx: mpsc::Sender<ToolRouterMessage>,
}

/// Cohesive actor group for tool orchestration + file watching.
pub struct ToolsSubsystem {
    pub deps: ToolsDeps,
}

impl Subsystem for ToolsSubsystem {
    type Handles = ToolsHandles;

    fn spawn(self, registry: &ServiceRegistry) -> Self::Handles {
        let (orch_tx, orch_rx) = mpsc::channel(64);
        let _ = registry.register::<ToolOrchestratorMessage>("tools.orchestrator", orch_tx.clone());
        let _join = ToolOrchestrator::new(
            self.deps.memory_graph_tx,
            self.deps.transport_tx,
            self.deps.mcp_tx,
            self.deps.llm_tx,
            self.deps.tool_router_tx,
        )
        .spawn(orch_rx);

        let (watcher_tx, watcher_rx) = mpsc::channel(64);
        let _ = registry.register::<FileWatcherMessage>("tools.watcher", watcher_tx.clone());
        let _join2 = FileWatcherActor::new().spawn(watcher_rx);

        ToolsHandles {
            orchestrator_tx: orch_tx,
            watcher_tx,
        }
    }

    fn actor_count(&self) -> usize {
        2
    }
}

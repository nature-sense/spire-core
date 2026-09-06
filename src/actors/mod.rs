// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Actor system — generic actors for the LLM/knowledge-store platform.
//!
//! Coding/development actors (build, project, planning, HAL, coordinator) live
//! in `spire-code`'s `actors` module and depend on this crate.

// Subsystem-owned actors live in `crate::subsystems::{graph,llm,tools,mcp,chat}`
// (each subsystem dir owns its actor implementation); this module re-exports
// their public types at the legacy flat paths plus the directly-spawned
// actors that are not part of a subsystem.

pub mod messages;
pub mod progress;
pub mod prompt_handler;
pub mod rag;
pub mod rag_ingest;
pub mod system_prompt;
pub mod tile;
pub mod tool_providers;
pub mod tools;
pub mod web_search;

// Re-export from the actor framework
pub use spire_actor::{Actor, ActorError, ActorSystem};

// Tool metadata + invocation message — originally in `spire-actor`, moved here
// so the runtime stays domain-agnostic.
pub use messages::{ToolInfo, ToolMessage};

// Re-export subsystem-owned actor types at the legacy flat paths.
pub use crate::subsystems::chat::chat::{ChatActor, ChatMessage};
pub use crate::subsystems::graph::memory_graph::{MemoryGraphActor, MemoryGraphMessage};
pub use crate::subsystems::llm::llm::{LlmActor, LlmConfig, LlmMessage};
pub use crate::subsystems::mcp::mcp_client::{McpClientActor, McpClientMessage};
pub use crate::subsystems::tools::file_watcher::{
    FileChangeKind, FileChangeNotification, FileEventBatch, FileEventInfo, FileWatcherActor,
    FileWatcherMessage,
};
pub use crate::subsystems::tools::tool_orchestrator::{ToolOrchestrator, ToolOrchestratorMessage};

// Directly-spawned actors.
pub use progress::{ProgressActor, ProgressMessage, ProgressStatus, ProgressUpdate};
pub use prompt_handler::{PromptContext, PromptHandlerActor, PromptHandlerMessage};
pub use rag::{RagActor, RagMessage};
pub use system_prompt::{SystemPromptActor, SystemPromptMessage};
pub use tile::{TileActor, TileFilters, TileMessage};
pub use tool_providers::{ToolRouterActor, ToolRouterMessage};
pub use tools::{ToolsActor, ToolsMessage};

// Core modules — static child actors providing general-purpose services.
pub use crate::modules::{
    FilesystemMessage, FilesystemModule, GitMessage, GitModule, ProcessMessage, ProcessModule,
    SearchMessage, SearchModule, TerminalMessage, TerminalModule,
};

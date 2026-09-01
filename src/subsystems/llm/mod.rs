// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! LLM subsystem — language-model client + prompt management.

pub mod llm;
pub use llm::{LlmActor, LlmConfig, LlmMessage};

use tokio::sync::mpsc;

use crate::actors::Actor;
use spire_actor::registry::ServiceRegistry;
use spire_actor::subsystem::Subsystem;

/// Handles for the LLM subsystem.
pub struct LlmHandles {
    /// Sender to the LLM actor.
    pub tx: mpsc::Sender<LlmMessage>,
}

/// Cohesive actor group for LLM/AI.
pub struct LlmSubsystem {
    pub config: LlmConfig,
}

impl Subsystem for LlmSubsystem {
    type Handles = LlmHandles;

    fn spawn(self, registry: &ServiceRegistry) -> Self::Handles {
        let (tx, rx) = mpsc::channel(64);
        let _ = registry.register::<LlmMessage>("llm.client", tx.clone());
        // Compatibility alias — existing FFI/main.rs consumers still resolve.
        let _ = registry.register::<LlmMessage>("llm", tx.clone());
        let _join = LlmActor::new(self.config).spawn(rx);
        LlmHandles { tx }
    }

    fn actor_count(&self) -> usize {
        1
    }
}
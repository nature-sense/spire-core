// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Chat subsystem — chat message handling.

pub mod chat;
pub use chat::{ChatActor, ChatMessage};

use tokio::sync::mpsc;

use crate::actors::Actor;
use spire_actor::registry::ServiceRegistry;
use spire_actor::subsystem::Subsystem;

/// Handles for the chat subsystem.
pub struct ChatHandles {
    /// Sender to the chat actor.
    pub tx: mpsc::Sender<ChatMessage>,
}

/// Cohesive actor group for chat.
pub struct ChatSubsystem;

impl Subsystem for ChatSubsystem {
    type Handles = ChatHandles;

    fn spawn(self, registry: &ServiceRegistry) -> Self::Handles {
        let (tx, rx) = mpsc::channel(64);
        let _ = registry.register::<ChatMessage>("chat.actor", tx.clone());
        // Compatibility alias.
        let _ = registry.register::<ChatMessage>("chat", tx.clone());
        let _join = ChatActor::new().spawn(rx);
        ChatHandles { tx }
    }

    fn actor_count(&self) -> usize {
        1
    }
}

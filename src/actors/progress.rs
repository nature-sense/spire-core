// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! ProgressActor — publishes progress updates and events.
//!
//! This actor manages progress tracking for long-running operations.
//! It broadcasts progress updates to subscribers (e.g., the transport layer
//! for forwarding to the VS Code extension as JSON-RPC notifications).

use async_trait::async_trait;
use tokio::sync::broadcast;

use crate::actors::Actor;

/// Status of a progress task.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ProgressStatus {
    Running,
    Completed,
    Failed,
}

/// A progress update event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProgressUpdate {
    pub task_id: String,
    pub message: String,
    pub percent: f64,
    pub status: ProgressStatus,
    /// Optional metadata (e.g., phase name for startup tracking).
    pub metadata: Option<serde_json::Value>,
}

/// Messages for the Progress actor.
pub enum ProgressMessage {
    /// Publish a progress update (broadcast to all subscribers).
    Publish { update: ProgressUpdate },
    /// Subscribe to progress updates.
    Subscribe {
        reply_to: tokio::sync::oneshot::Sender<broadcast::Receiver<ProgressUpdate>>,
    },
}

/// Actor that manages progress tracking and event broadcasting.
pub struct ProgressActor {
    /// Broadcast channel for progress updates.
    tx: broadcast::Sender<ProgressUpdate>,
}

impl ProgressActor {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(256);
        Self { tx }
    }
}

impl Default for ProgressActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for ProgressActor {
    type Message = ProgressMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            ProgressMessage::Publish { update } => {
                let _ = self.tx.send(update);
            }
            ProgressMessage::Subscribe { reply_to } => {
                let rx = self.tx.subscribe();
                let _ = reply_to.send(rx);
            }
        }
    }
}

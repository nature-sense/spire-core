// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Graph subsystem — persistence for the knowledge graph.
//!
//! Owns the `MemoryGraphActor` (graph database, embeddings, snapshots).
//! Other subsystems reach the graph exclusively through the registry key
//! `"graph.memory"` (compatibility alias `"memory_graph"`).

pub mod memory_graph;
pub use memory_graph::{MemoryGraphActor, MemoryGraphMessage};

use tokio::sync::mpsc;

use spire_actor::registry::ServiceRegistry;
use spire_actor::subsystem::Subsystem;
use spire_actor::Actor;

/// Handles for the graph subsystem (its message senders).
pub struct GraphHandles {
    /// Sender to the memory-graph actor.
    pub tx: mpsc::Sender<MemoryGraphMessage>,
}

/// Cohesive actor group for graph persistence.
pub struct GraphSubsystem;

impl Subsystem for GraphSubsystem {
    type Handles = GraphHandles;

    fn spawn(self, registry: &ServiceRegistry) -> Self::Handles {
        let (tx, rx) = mpsc::channel(64);
        let _ = registry.register::<MemoryGraphMessage>("graph.memory", tx.clone());
        // Compatibility alias — existing FFI/main.rs consumers still resolve.
        let _ = registry.register::<MemoryGraphMessage>("memory_graph", tx.clone());
        let _join = MemoryGraphActor::new().spawn(rx);
        GraphHandles { tx }
    }

    fn actor_count(&self) -> usize {
        1
    }
}

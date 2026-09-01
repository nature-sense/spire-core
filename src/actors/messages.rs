// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Tool-related message types shared across the platform's actors.
//!
//! These types originally lived in `spire-actor`'s `messages` module. They were
//! removed from the runtime to keep it domain-agnostic, so they now live here in
//! `spire-core` (the AI-app platform). The canonical definitions remain in
//! `naturesense/tools/spire-app/crates/spire-actor/src/messages.rs`.

use serde_json::Value;
use spire_actor::Responder;

/// A generic tool invocation message.
pub struct ToolMessage {
    pub tool: String,
    pub args: Value,
    pub response_tx: Responder<Value>,
}

/// Metadata describing a registered tool.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

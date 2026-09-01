// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! ToolRegistry — a registration service for tools.
//!
//! Tool ROUTING lives here (generic); tool DEFINITIONS live with their owners.
//! Nothing is registered by default — the app composer registers the full set
//! at startup: generic tools from `spire-core` (VS Code extension, in-process
//! core modules, web search, RAG, MCP) and coding tools from `spire-code`
//! (project_*, build). Registered handlers are type-erased closures, so the
//! registry never sees a concrete actor's message type.

use crate::actors::ToolInfo;
use futures::future::BoxFuture;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Type-erased async tool call handler: takes the tool's JSON args and returns
/// a JSON result (or an error string).
pub type ToolHandler =
    Arc<dyn Fn(Value) -> BoxFuture<'static, Result<Value, String>> + Send + Sync>;

/// Thread-safe registry of tools, keyed by exact tool name.
///
/// The registry is intentionally empty by default: the app composer calls
/// `register`/`register_many` to populate it with the tools each layer owns.
#[derive(Default)]
pub struct ToolRegistry {
    tools: Mutex<HashMap<String, (ToolInfo, ToolHandler)>>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a single tool. Returns `Err` if the name is already taken.
    pub fn register(&self, info: ToolInfo, handler: ToolHandler) -> Result<(), String> {
        let mut map = self
            .tools
            .lock()
            .map_err(|e| format!("tool registry poisoned: {e}"))?;
        if map.contains_key(&info.name) {
            return Err(format!("Tool '{}' already registered", info.name));
        }
        map.insert(info.name.clone(), (info, handler));
        Ok(())
    }

    /// Register many tools at once; fails fast on the first duplicate.
    pub fn register_many(&self, tools: Vec<(ToolInfo, ToolHandler)>) -> Result<(), String> {
        for (info, handler) in tools {
            self.register(info, handler)?;
        }
        Ok(())
    }

    /// All registered tool metadata.
    pub fn list(&self) -> Vec<ToolInfo> {
        self.tools
            .lock()
            .map(|m| m.values().map(|(info, _)| info.clone()).collect())
            .unwrap_or_default()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Call a registered tool by exact name. Returns `None` if unknown.
    pub fn call(
        &self,
        name: &str,
        args: Value,
    ) -> Option<BoxFuture<'static, Result<Value, String>>> {
        let handler = self.tools.lock().ok()?.get(name).map(|(_, h)| h.clone())?;
        Some(handler(args))
    }
}

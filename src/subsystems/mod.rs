// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Actor subsystems — cohesive actor groups by functional domain.
//!
//! Each subsystem owns the actors in its domain, spawns them, and registers
//! their senders in the parent system's `ServiceRegistry`. Migration happens
//! one subsystem at a time (graph → llm → tools → mcp → build → project →
//! planning → chat) with `actors/mod.rs` re-exporting old paths meanwhile.

pub mod graph;
pub mod llm;
pub mod chat;
pub mod mcp;
pub mod tools;
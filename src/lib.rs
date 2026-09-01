// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Spire Core — Rust actor system with MCP client.
//!
//! This library provides the core functionality for the Spire VS Code extension:
//! - Actor framework for message-passing concurrency
//! - MCP client for connecting to external MCP servers
//! - JSON-RPC 2.0 transport over stdin/stdout
//! - Chat dialog management
//! - Tool registry and dispatch
//! - Knowledge graph with SeleneDB-backed storage
//! - Text embedding with Candle (all-MiniLM-L6-v2)
//!
//! The library can be used as a standalone binary (`spire-core`) or as a
//! library for testing.

pub mod actors;
pub mod analyzer;
pub mod build_types;
pub mod config;
pub mod embedder;
pub mod graph;
pub mod mcp;
pub mod models;
pub mod modules;
pub mod platform;
pub mod subsystems;
pub mod transport;
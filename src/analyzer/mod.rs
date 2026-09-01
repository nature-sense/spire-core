// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Project Analyzer — standalone library for analysing project directory structure.
//!
//! This module provides the building blocks for project analysis:
//!
//! 1. **Scanner** — Walk the filesystem, respecting `.gitignore`, collecting `FileInfo`.
//! 2. **Tree builder** — Assemble the flat file list into a hierarchical
//!    `DirectoryNode` tree with language detection, role classification, and
//!    line estimation.
//!
//! The top-level orchestration is handled by the `ProjectAnalyzerActor`, which
//! delegates build analysis to external MCP servers. Build-system-specific
//! parsing lives exclusively in MCP servers (mcp-cargo, mcp-node, etc.), not
//! in this crate.

pub mod models;
pub mod scanner;
pub mod tree_builder;

pub use models::*;
pub use scanner::*;
pub use tree_builder::*;

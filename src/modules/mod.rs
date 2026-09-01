// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Core modules — static child actors providing general-purpose services.
//!
//! Unlike `build_modules` (which share the `BuildModuleMessage` protocol),
//! each core module has its own message enum and interface:
//!
//! - `filesystem` — read/write/list/delete/move/copy files
//! - `git`        — status/diff/log/branch/commit
//! - `process`    — spawn/kill/list/monitor subprocesses
//! - `search`     — file/content search
//! - `terminal`   — execute commands, capture output
//!
//! Each is a long-lived top-level `Actor` spawned once at startup and
//! registered in the `ServiceRegistry` under a named key (e.g. "filesystem",
//! "git").

pub mod filesystem;
pub use filesystem::{FilesystemMessage, FilesystemModule};

pub mod git;
pub use git::{GitMessage, GitModule};

pub mod process;
pub use process::{ProcessMessage, ProcessModule};

pub mod search;
pub use search::{SearchMessage, SearchModule};

pub mod terminal;
pub use terminal::{TerminalMessage, TerminalModule};

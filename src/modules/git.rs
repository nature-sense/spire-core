// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Git module — status/diff/log/branch/commit via the git CLI.

use async_trait::async_trait;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tokio::sync::oneshot;

use spire_actor::Actor;

/// Messages for the Git module.
pub enum GitMessage {
    /// Show working tree status.
    Status {
        path: PathBuf,
        reply_to: oneshot::Sender<Result<String, String>>,
    },
    /// Show diff (working tree or staged).
    Diff {
        path: PathBuf,
        staged: bool,
        reply_to: oneshot::Sender<Result<String, String>>,
    },
    /// Show recent commit log.
    Log {
        path: PathBuf,
        count: usize,
        reply_to: oneshot::Sender<Result<String, String>>,
    },
    /// List branches.
    Branch {
        path: PathBuf,
        reply_to: oneshot::Sender<Result<String, String>>,
    },
    /// Commit staged changes with a message.
    Commit {
        path: PathBuf,
        message: String,
        reply_to: oneshot::Sender<Result<String, String>>,
    },
    /// LLM tool invocation.
    CallTool {
        tool_name: String,
        args: serde_json::Value,
        reply_to: oneshot::Sender<serde_json::Value>,
    },
    /// List this module's registered tools (actor-based discovery).
    ListTools {
        reply_to: oneshot::Sender<Vec<crate::actors::ToolInfo>>,
    },
}

/// Static Git module.
pub struct GitModule;

impl GitModule {
    pub fn new() -> Self {
        Self
    }

    /// Run `git <args>` in a directory, returning combined stdout/stderr.
    async fn git_run(&self, path: &PathBuf, args: &[&str]) -> Result<String, String> {
        let mut cmd = Command::new("git");
        cmd.current_dir(path)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let output = cmd
            .output()
            .await
            .map_err(|e| format!("Failed to execute git: {e}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let text = if stderr.is_empty() {
            stdout.to_string()
        } else {
            format!("{stdout}\n{stderr}")
        };
        if output.status.success() {
            Ok(text.trim().to_string())
        } else {
            Err(text)
        }
    }

    pub async fn status(&self, path: &PathBuf) -> Result<String, String> {
        self.git_run(path, &["status", "--short"]).await
    }

    pub async fn diff(&self, path: &PathBuf, staged: bool) -> Result<String, String> {
        if staged {
            self.git_run(path, &["diff", "--cached"]).await
        } else {
            self.git_run(path, &["diff"]).await
        }
    }

    pub async fn log(&self, path: &PathBuf, count: usize) -> Result<String, String> {
        let n = count.to_string();
        self.git_run(path, &["log", "--oneline", "-n", &n]).await
    }

    pub async fn branch(&self, path: &PathBuf) -> Result<String, String> {
        self.git_run(path, &["branch", "-a"]).await
    }

    pub async fn commit(&self, path: &PathBuf, message: &str) -> Result<String, String> {
        self.git_run(path, &["commit", "-m", message]).await
    }

    /// Dispatch an LLM tool call by name (async: git commands are async).
    async fn call_tool(&self, tool_name: &str, args: serde_json::Value) -> serde_json::Value {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => PathBuf::from(p),
            None => return serde_json::json!({ "error": "missing 'path' arg" }),
        };
        let result = match tool_name {
            "git_status" => self.status(&path).await,
            "git_diff" => {
                let staged = args
                    .get("staged")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                self.diff(&path, staged).await
            }
            "git_log" => {
                let count = args
                    .get("count")
                    .and_then(|v| v.as_u64())
                    .map(|c| c as usize)
                    .unwrap_or(10);
                self.log(&path, count).await
            }
            "git_branch" => self.branch(&path).await,
            "git_commit" => {
                let msg = args.get("message").and_then(|v| v.as_str()).unwrap_or("");
                self.commit(&path, msg).await
            }
            other => Err(format!("Unknown git tool: {other}")),
        };
        match result {
            Ok(output) => serde_json::json!({ "output": output }),
            Err(e) => serde_json::json!({ "error": e }),
        }
    }

    /// Build this module's tool list.
    fn list_tools(&self) -> Vec<crate::actors::ToolInfo> {
        vec![
            crate::actors::ToolInfo {
                name: "git_status".to_string(),
                description: "Show git working tree status (short format).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
            },
            crate::actors::ToolInfo {
                name: "git_diff".to_string(),
                description: "Show git diff (working tree or staged).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "staged": { "type": "boolean" }
                    },
                    "required": ["path"]
                }),
            },
            crate::actors::ToolInfo {
                name: "git_log".to_string(),
                description: "Show recent git commit log (oneline).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "count": { "type": "integer" }
                    },
                    "required": ["path"]
                }),
            },
            crate::actors::ToolInfo {
                name: "git_branch".to_string(),
                description: "List local + remote git branches.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
            },
            crate::actors::ToolInfo {
                name: "git_commit".to_string(),
                description: "Commit staged changes with a message.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "message": { "type": "string" }
                    },
                    "required": ["path", "message"]
                }),
            },
        ]
    }
}

impl Default for GitModule {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for GitModule {
    type Message = GitMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            GitMessage::Status { path, reply_to } => {
                let _ = reply_to.send(self.status(&path).await);
            }
            GitMessage::Diff {
                path,
                staged,
                reply_to,
            } => {
                let _ = reply_to.send(self.diff(&path, staged).await);
            }
            GitMessage::Log {
                path,
                count,
                reply_to,
            } => {
                let _ = reply_to.send(self.log(&path, count).await);
            }
            GitMessage::Branch { path, reply_to } => {
                let _ = reply_to.send(self.branch(&path).await);
            }
            GitMessage::Commit {
                path,
                message,
                reply_to,
            } => {
                let _ = reply_to.send(self.commit(&path, &message).await);
            }
            GitMessage::CallTool {
                tool_name,
                args,
                reply_to,
            } => {
                let _ = reply_to.send(self.call_tool(&tool_name, args).await);
            }
            GitMessage::ListTools { reply_to } => {
                let _ = reply_to.send(self.list_tools());
            }
        }
    }
}

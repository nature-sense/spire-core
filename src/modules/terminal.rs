// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Terminal module — execute commands and manage output sessions.
//!
//! Executes a command and stores its output in a session map (the actor's
//! private state). Callers can retrieve output later by session ID.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::process::Command;
use tokio::sync::oneshot;

use spire_actor::Actor;

/// A terminal session recording a command's output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalSession {
    pub id: String,
    pub command: String,
    pub exit_code: Option<i32>,
    pub output: String,
    pub recorded_at: String,
}

/// Messages for the Terminal module.
pub enum TerminalMessage {
    /// Execute a command in a working dir; store output under a new session.
    Execute {
        command: String,
        working_dir: PathBuf,
        reply_to: oneshot::Sender<Result<TerminalSession, String>>,
    },
    /// Fetch a session's recorded output by ID.
    GetOutput {
        session_id: String,
        reply_to: oneshot::Sender<Option<TerminalSession>>,
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

/// Static Terminal module.
pub struct TerminalModule {
    /// Recorded command sessions keyed by ID.
    sessions: HashMap<String, TerminalSession>,
}

impl TerminalModule {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    async fn execute(
        &mut self,
        command: &str,
        working_dir: &PathBuf,
    ) -> Result<TerminalSession, String> {
        // Split command into program + args (simple whitespace split).
        let mut parts = command.split_whitespace();
        let program = parts.next().ok_or_else(|| "Empty command".to_string())?;
        let args: Vec<String> = parts.map(|s| s.to_string()).collect();

        let mut cmd = Command::new(program);
        cmd.args(&args);
        if !working_dir.as_os_str().is_empty() {
            cmd.current_dir(working_dir);
        }
        let output = cmd
            .output()
            .await
            .map_err(|e| format!("Failed to execute {program}: {e}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let id = format!(
            "session-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        );
        let session = TerminalSession {
            id: id.clone(),
            command: command.to_string(),
            exit_code: output.status.code(),
            output: if stderr.is_empty() {
                stdout
            } else {
                format!("{stdout}\n{stderr}")
            },
            recorded_at: format!("{:?}", SystemTime::now()),
        };
        self.sessions.insert(id, session.clone());
        Ok(session)
    }

    /// Dispatch an LLM tool call by name.
    async fn call_tool(&mut self, tool_name: &str, args: serde_json::Value) -> serde_json::Value {
        match tool_name {
            "terminal_execute" => {
                let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
                if command.is_empty() {
                    return serde_json::json!({ "error": "missing 'command' arg" });
                }
                let wd = args
                    .get("working_dir")
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from)
                    .unwrap_or_default();
                match self.execute(command, &wd).await {
                    Ok(session) => serde_json::to_value(session)
                        .unwrap_or(serde_json::json!({ "error": "serialize" })),
                    Err(e) => serde_json::json!({ "error": e }),
                }
            }
            "terminal_get_output" => {
                let id = args
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                serde_json::to_value(self.sessions.get(id).cloned())
                    .unwrap_or(serde_json::json!(null))
            }
            other => serde_json::json!({ "error": format!("Unknown terminal tool: {other}") }),
        }
    }

    /// Build this module's tools.
    fn list_tools(&self) -> Vec<crate::actors::ToolInfo> {
        vec![
            crate::actors::ToolInfo {
                name: "terminal_execute".to_string(),
                description: "Execute a command and capture its output into a session.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" },
                        "working_dir": { "type": "string" }
                    },
                    "required": ["command"]
                }),
            },
            crate::actors::ToolInfo {
                name: "terminal_get_output".to_string(),
                description: "Fetch a recorded terminal session's output by its session_id."
                    .to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "session_id": { "type": "string" } },
                    "required": ["session_id"]
                }),
            },
        ]
    }
}

impl Default for TerminalModule {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for TerminalModule {
    type Message = TerminalMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            TerminalMessage::Execute {
                command,
                working_dir,
                reply_to,
            } => {
                let _ = reply_to.send(self.execute(&command, &working_dir).await);
            }
            TerminalMessage::GetOutput {
                session_id,
                reply_to,
            } => {
                let _ = reply_to.send(self.sessions.get(&session_id).cloned());
            }
            TerminalMessage::CallTool {
                tool_name,
                args,
                reply_to,
            } => {
                let _ = reply_to.send(self.call_tool(&tool_name, args).await);
            }
            TerminalMessage::ListTools { reply_to } => {
                let _ = reply_to.send(self.list_tools());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn execute_and_retrieve_session() {
        let mut module = TerminalModule::new();
        let session = module.execute("echo hello", &PathBuf::new()).await.unwrap();
        assert!(session.output.contains("hello"));
        assert_eq!(session.exit_code, Some(0));

        let fetched = module.sessions.get(&session.id).cloned();
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().command, "echo hello");
    }
}

// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Process module — spawn/kill/list/wait on subprocesses.
//!
//! Maintains a private map of spawned children keyed by a monotonic ID.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::process::Child;
use tokio::sync::oneshot;

use spire_actor::Actor;

/// A running process descriptor returned to callers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    pub id: u32,
    pub command: String,
    pub args: Vec<String>,
    pub working_dir: String,
}

/// Messages for the Process module.
pub enum ProcessMessage {
    /// Spawn a subprocess and track it.
    Spawn {
        command: String,
        args: Vec<String>,
        working_dir: PathBuf,
        reply_to: oneshot::Sender<Result<ProcessInfo, String>>,
    },
    /// Kill a tracked process by ID.
    Kill {
        id: u32,
        reply_to: oneshot::Sender<Result<(), String>>,
    },
    /// List all tracked processes.
    List {
        reply_to: oneshot::Sender<Vec<ProcessInfo>>,
    },
    /// Wait for a process to exit and return its exit code.
    Wait {
        id: u32,
        reply_to: oneshot::Sender<Result<Option<i32>, String>>,
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

/// Static Process module.
pub struct ProcessModule {
    /// Tracked running children keyed by a monotonically increasing ID.
    processes: HashMap<u32, Child>,
    /// Unique ID counter.
    next_id: u32,
}

impl ProcessModule {
    pub fn new() -> Self {
        Self {
            processes: HashMap::new(),
            next_id: 1,
        }
    }

    fn spawn_process(
        &mut self,
        command: &str,
        args: &[String],
        working_dir: &PathBuf,
    ) -> Result<ProcessInfo, String> {
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args);
        if !working_dir.as_os_str().is_empty() {
            cmd.current_dir(working_dir);
        }
        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn {command}: {e}"))?;

        let id = self.next_id;
        self.next_id += 1;
        self.processes.insert(id, child);

        Ok(ProcessInfo {
            id,
            command: command.to_string(),
            args: args.to_vec(),
            working_dir: working_dir.to_string_lossy().to_string(),
        })
    }

    async fn wait(&mut self, id: u32) -> Result<Option<i32>, String> {
        match self.processes.remove(&id) {
            Some(mut child) => child
                .wait()
                .await
                .map(|s| s.code())
                .map_err(|e| format!("Wait failed for process {id}: {e}")),
            None => Err(format!("No process with id {id}")),
        }
    }

    /// Dispatch an LLM tool call by name.
    async fn call_tool(&mut self, tool_name: &str, args: serde_json::Value) -> serde_json::Value {
        match tool_name {
            "process_spawn" => {
                let command = args
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if command.is_empty() {
                    return serde_json::json!({ "error": "missing 'command' arg" });
                }
                let a = args
                    .get("args")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let arg_strings: Vec<String> = a
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                let wd = args
                    .get("working_dir")
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from)
                    .unwrap_or_default();
                match self.spawn_process(&command, &arg_strings, &wd) {
                    Ok(info) => serde_json::to_value(info)
                        .unwrap_or(serde_json::json!({ "error": "serialize" })),
                    Err(e) => serde_json::json!({ "error": e }),
                }
            }
            "process_kill" | "process_wait" => {
                let id = match args.get("id").and_then(|v| v.as_u64()) {
                    Some(i) => i as u32,
                    None => return serde_json::json!({ "error": "missing 'id' arg" }),
                };
                let result = if tool_name == "process_kill" {
                    match self.processes.remove(&id) {
                        Some(mut child) => child
                            .kill()
                            .await
                            .map(|_| serde_json::json!({ "ok": true }))
                            .map_err(|e| format!("Kill failed: {e}")),
                        None => Err(format!("No process with id {id}")),
                    }
                } else {
                    self.wait(id)
                        .await
                        .map(|code| serde_json::json!({ "exit_code": code }))
                };
                match result {
                    Ok(_) => serde_json::json!({ "ok": true }),
                    Err(e) => serde_json::json!({ "error": e }),
                }
            }
            "process_list" => {
                let info: Vec<ProcessInfo> = self
                    .processes
                    .iter()
                    .map(|(id, child)| ProcessInfo {
                        id: *id,
                        command: child
                            .id()
                            .map(|p| p.to_string())
                            .unwrap_or_else(|| "unknown".to_string()),
                        args: vec![],
                        working_dir: String::new(),
                    })
                    .collect();
                serde_json::json!(info)
            }
            other => serde_json::json!({ "error": format!("Unknown process tool: {other}") }),
        }
    }

    /// Build this module's tool list.
    fn list_tools(&self) -> Vec<crate::actors::ToolInfo> {
        vec![
            crate::actors::ToolInfo {
                name: "process_spawn".to_string(),
                description: "Spawn a subprocess and track it.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" },
                        "args": { "type": "array", "items": { "type": "string" } },
                        "working_dir": { "type": "string" }
                    },
                    "required": ["command"]
                }),
            },
            crate::actors::ToolInfo {
                name: "process_kill".to_string(),
                description: "Kill a tracked process by its numeric ID.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "id": { "type": "integer" } },
                    "required": ["id"]
                }),
            },
            crate::actors::ToolInfo {
                name: "process_wait".to_string(),
                description: "Wait for a tracked process to exit and return its exit code."
                    .to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "id": { "type": "integer" } },
                    "required": ["id"]
                }),
            },
            crate::actors::ToolInfo {
                name: "process_list".to_string(),
                description: "List all tracked processes.".to_string(),
                input_schema: serde_json::json!({ "type": "object", "properties": {} }),
            },
        ]
    }
}

impl Default for ProcessModule {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for ProcessModule {
    type Message = ProcessMessage;


    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            ProcessMessage::Spawn {
                command,
                args,
                working_dir,
                reply_to,
            } => {
                let _ = reply_to.send(self.spawn_process(&command, &args, &working_dir));
            }
            ProcessMessage::Kill { id, reply_to } => {
                let result = match self.processes.remove(&id) {
                    Some(mut child) => match child.kill().await {
                        Ok(_) => Ok(()),
                        Err(e) => Err(format!("Kill failed for process {id}: {e}")),
                    },
                    None => Err(format!("No process with id {id}")),
                };
                let _ = reply_to.send(result);
            }
            ProcessMessage::List { reply_to } => {
                let info: Vec<ProcessInfo> = self
                    .processes
                    .iter()
                    .map(|(id, child)| ProcessInfo {
                        id: *id,
                        command: child
                            .id()
                            .map(|pid| pid.to_string())
                            .unwrap_or_else(|| "unknown".to_string()),
                        args: vec![],
                        working_dir: String::new(),
                    })
                    .collect();
                let _ = reply_to.send(info);
            }
            ProcessMessage::Wait { id, reply_to } => {
                let _ = reply_to.send(self.wait(id).await);
            }
            ProcessMessage::CallTool {
                tool_name,
                args,
                reply_to,
            } => {
                let _ = reply_to.send(self.call_tool(&tool_name, args).await);
            }
            ProcessMessage::ListTools { reply_to } => {
                let _ = reply_to.send(self.list_tools());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn spawn_and_wait_echo() {
        let mut module = ProcessModule::new();
        let info = module
            .spawn_process("echo", &["hi".to_string()], &PathBuf::new())
            .unwrap();
        assert_eq!(info.command, "echo");
        let code = module.wait(info.id).await.unwrap();
        assert_eq!(code, Some(0));
    }
}

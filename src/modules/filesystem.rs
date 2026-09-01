// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Filesystem module — read/write/list/delete/move/copy files, plus a strict
//! `apply_patch` unified-diff tool (apply-or-fail).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::sync::oneshot;

use spire_actor::Actor;

/// A single listing entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub name: String,
    pub path: String,
    pub is_directory: bool,
    pub is_symlink: bool,
    pub size: u64,
}

/// Messages for the Filesystem module.
pub enum FilesystemMessage {
    ReadFile {
        path: PathBuf,
        reply_to: oneshot::Sender<Result<String, String>>,
    },
    WriteFile {
        path: PathBuf,
        content: String,
        reply_to: oneshot::Sender<Result<(), String>>,
    },
    ListDirectory {
        path: PathBuf,
        reply_to: oneshot::Sender<Result<Vec<FileEntry>, String>>,
    },
    Delete {
        path: PathBuf,
        reply_to: oneshot::Sender<Result<(), String>>,
    },
    Move {
        from: PathBuf,
        to: PathBuf,
        reply_to: oneshot::Sender<Result<(), String>>,
    },
    Copy {
        from: PathBuf,
        to: PathBuf,
        reply_to: oneshot::Sender<Result<(), String>>,
    },
    /// Apply a strict, context-verified unified diff to a file.
    ApplyPatch {
        path: PathBuf,
        patch: String,
        reply_to: oneshot::Sender<serde_json::Value>,
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

/// Static Filesystem module.
pub struct FilesystemModule;

fn copy_dir_recursive(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let src = entry.path();
        let dst = to.join(entry.file_name());
        if src.is_dir() {
            copy_dir_recursive(&src, &dst)?;
        } else {
            fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

impl FilesystemModule {
    pub fn new() -> Self {
        Self
    }

    fn read_file(&self, path: &PathBuf) -> Result<String, String> {
        fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))
    }

    fn write_file(&self, path: &PathBuf, content: &str) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create dirs {}: {e}", parent.display()))?;
        }
        fs::write(path, content).map_err(|e| format!("Failed to write {}: {e}", path.display()))
    }

    fn list_directory(&self, path: &PathBuf) -> Result<Vec<FileEntry>, String> {
        let entries = fs::read_dir(path)
            .map_err(|e| format!("Failed to read dir {}: {e}", path.display()))?;
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| format!("Dir entry error: {e}"))?;
            let meta = entry
                .metadata()
                .map_err(|e| format!("Metadata error: {e}"))?;
            out.push(FileEntry {
                name: entry.file_name().to_string_lossy().to_string(),
                path: entry.path().to_string_lossy().to_string(),
                is_directory: meta.is_dir(),
                is_symlink: meta.file_type().is_symlink(),
                size: meta.len(),
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    fn delete(&self, path: &PathBuf) -> Result<(), String> {
        let meta = fs::metadata(path).map_err(|e| format!("Stat {}: {e}", path.display()))?;
        if meta.is_dir() {
            fs::remove_dir_all(path)
                .map_err(|e| format!("Failed to remove dir {}: {e}", path.display()))
        } else {
            fs::remove_file(path)
                .map_err(|e| format!("Failed to remove file {}: {e}", path.display()))
        }
    }

    fn move_path(&self, from: &PathBuf, to: &PathBuf) -> Result<(), String> {
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::rename(from, to).map_err(|e| format!("Failed to move {:?} to {:?}: {e}", from, to))
    }

    fn copy_path(&self, from: &PathBuf, to: &PathBuf) -> Result<(), String> {
        let meta = fs::metadata(from).map_err(|e| format!("Stat {}: {e}", from.display()))?;
        if meta.is_dir() {
            copy_dir_recursive(from, to).map_err(|e| e.to_string())
        } else {
            if let Some(parent) = to.parent() {
                fs::create_dir_all(parent).ok();
            }
            fs::copy(from, to)
                .map_err(|e| format!("Failed to copy {:?} to {:?}: {e}", from, to))?;
            Ok(())
        }
    }

    /// Dispatch an LLM tool call by name.
    fn call_tool(&self, tool_name: &str, args: serde_json::Value) -> serde_json::Value {
        macro_rules! path_arg {
            () => {
                match args.get("path").and_then(|v| v.as_str()) {
                    Some(p) => PathBuf::from(p),
                    None => return serde_json::json!({ "error": "missing 'path' arg" }),
                }
            };
        }
        macro_rules! path_arg2 {
            ($name:literal) => {
                match args.get($name).and_then(|v| v.as_str()) {
                    Some(p) => PathBuf::from(p),
                    None => {
                        return serde_json::json!({ "error": concat!("missing '", $name, "' arg") })
                    }
                }
            };
        }
        match tool_name {
            "filesystem_read" => {
                let path = path_arg!();
                serde_json::to_value(self.read_file(&path))
                    .unwrap_or(serde_json::json!({ "error": "serialize" }))
            }
            "filesystem_write" => {
                let path = path_arg!();
                let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
                serde_json::to_value(self.write_file(&path, content))
                    .unwrap_or(serde_json::json!({ "error": "serialize" }))
            }
            "filesystem_list" => {
                let path = path_arg!();
                serde_json::to_value(self.list_directory(&path))
                    .unwrap_or(serde_json::json!({ "error": "serialize" }))
            }
            "filesystem_delete" => {
                let path = path_arg!();
                serde_json::to_value(self.delete(&path))
                    .unwrap_or(serde_json::json!({ "error": "serialize" }))
            }
            "filesystem_move" | "filesystem_copy" => {
                let from = path_arg2!("from");
                let to = path_arg2!("to");
                let result = if tool_name == "filesystem_move" {
                    self.move_path(&from, &to)
                } else {
                    self.copy_path(&from, &to)
                };
                serde_json::to_value(result).unwrap_or(serde_json::json!({ "error": "serialize" }))
            }
            "apply_patch" => {
                let path = path_arg!();
                let patch = args.get("patch").and_then(|v| v.as_str()).unwrap_or("");
                apply_unified_patch(&path, patch)
            }
            other => serde_json::json!({ "error": format!("Unknown filesystem tool: {other}") }),
        }
    }

    /// Build this module's tool list.
    fn list_tools(&self) -> Vec<crate::actors::ToolInfo> {
        vec![
            crate::actors::ToolInfo {
                name: "filesystem_read".to_string(),
                description: "Read a file's contents as UTF-8 text.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
            },
            crate::actors::ToolInfo {
                name: "filesystem_write".to_string(),
                description: "Write UTF-8 content to a file, creating parent dirs.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    },
                    "required": ["path", "content"]
                }),
            },
            crate::actors::ToolInfo {
                name: "filesystem_list".to_string(),
                description: "List a directory's immediate entries.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
            },
            crate::actors::ToolInfo {
                name: "filesystem_delete".to_string(),
                description: "Delete a file or directory (recursive for dirs).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
            },
            crate::actors::ToolInfo {
                name: "filesystem_move".to_string(),
                description: "Move/rename a file or directory.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "from": { "type": "string" },
                        "to": { "type": "string" }
                    },
                    "required": ["from", "to"]
                }),
            },
            crate::actors::ToolInfo {
                name: "filesystem_copy".to_string(),
                description: "Copy a file or directory (recursive).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "from": { "type": "string" },
                        "to": { "type": "string" }
                    },
                    "required": ["from", "to"]
                }),
            },
            crate::actors::ToolInfo {
                name: "apply_patch".to_string(),
                description: "Apply a strict, context-verified unified diff to a file. Every context/deleted line must match the file exactly or the whole patch is rejected (apply-or-fail) — the file is never corrupted by stale edits. Use this instead of writing whole files for small, surgical changes.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute path to the file to patch" },
                        "patch": { "type": "string", "description": "Unified diff text (hunks of the form '@@ -old +new @@' with context/'-'/'+' lines)" }
                    },
                    "required": ["path", "patch"]
                }),
            },
        ]
    }
}

impl Default for FilesystemModule {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for FilesystemModule {
    type Message = FilesystemMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            FilesystemMessage::ReadFile { path, reply_to } => {
                let _ = reply_to.send(self.read_file(&path));
            }
            FilesystemMessage::WriteFile {
                path,
                content,
                reply_to,
            } => {
                let _ = reply_to.send(self.write_file(&path, &content));
            }
            FilesystemMessage::ListDirectory { path, reply_to } => {
                let _ = reply_to.send(self.list_directory(&path));
            }
            FilesystemMessage::Delete { path, reply_to } => {
                let _ = reply_to.send(self.delete(&path));
            }
            FilesystemMessage::Move { from, to, reply_to } => {
                let _ = reply_to.send(self.move_path(&from, &to));
            }
            FilesystemMessage::Copy { from, to, reply_to } => {
                let _ = reply_to.send(self.copy_path(&from, &to));
            }
            FilesystemMessage::ApplyPatch { path, patch, reply_to } => {
                let _ = reply_to.send(apply_unified_patch(&path, &patch));
            }
            FilesystemMessage::CallTool {
                tool_name,
                args,
                reply_to,
            } => {
                let _ = reply_to.send(self.call_tool(&tool_name, args));
            }
            FilesystemMessage::ListTools { reply_to } => {
                let _ = reply_to.send(self.list_tools());
            }
        }
    }
}

/// Apply a strict, context-verified unified diff to `file_path`.
///
/// Supports hunks of the form:
/// ```text
/// @@ -old_start,old_count +new_start,new_count @@
///  context line
/// -deleted line
/// +added line
/// ```
///
/// Every context (` `) and deleted (`-`) line must match the file exactly at
/// its position. On any mismatch the whole patch is rejected (apply-or-fail)
/// and the file is left untouched — this is what prevents stale/offset edits
/// from corrupting files. Returns `{"ok": true}` on success, otherwise
/// `{"ok": false, "error": "..."}`.
pub fn apply_unified_patch(file_path: &Path, patch: &str) -> serde_json::Value {
    let original = match fs::read_to_string(file_path) {
        Ok(s) => s,
        Err(e) => return serde_json::json!({ "ok": false, "error": format!("read failed: {e}") }),
    };
    // Owned lines — mixing borrowed (original) and hunk-owned strings in one
    // backing Vec would dangle when hunk locals drop. Own everything.
    let mut lines: Vec<String> = original.lines().map(|s| s.to_string()).collect();

    let patch_lines: Vec<&str> = patch.lines().collect();
    let mut i = 0usize;
    // Skip prologue (e.g. "--- a/file" / "+++ b/file").
    while i < patch_lines.len() && !patch_lines[i].starts_with("@@") {
        i += 1;
    }

    while i < patch_lines.len() {
        let header = patch_lines[i];
        if !header.starts_with("@@ ") {
            return serde_json::json!({ "ok": false, "error": format!("expected @@ hunk header, got: {header}") });
        }
        // Parse "@@ -old_start[,old_count] +new_start[,new_count] @@".
        let body = &header[3..]; // strip "@@ "
        let (old_side, new_part) = body.split_once(" +").unwrap_or((body, ""));
        let old_start: i64 = old_side
            .trim_start_matches('-')
            .split(',')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let new_part_clean = new_part.split(" @@").next().unwrap_or("");
        let _new_start: i64 = new_part_clean
            .trim_start_matches('+')
            .split(',')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        i += 1;

        // Cursor into `lines` for this hunk's old side (0-based).
        let mut cursor: i64 = if old_start == 0 { 0 } else { old_start - 1 };
        let mut new_lines: Vec<String> = Vec::new();

        while i < patch_lines.len() && !patch_lines[i].starts_with("@@") {
            let line = patch_lines[i];
            i += 1;
            if let Some(ctx) = line.strip_prefix(' ') {
                if cursor >= 0
                    && (cursor as usize) < lines.len()
                    && lines[cursor as usize] == *ctx
                {
                    new_lines.push(ctx.to_string());
                    cursor += 1;
                } else {
                    return serde_json::json!({ "ok": false, "error": "context mismatch" });
                }
            } else if let Some(del) = line.strip_prefix('-') {
                if cursor >= 0
                    && (cursor as usize) < lines.len()
                    && lines[cursor as usize] == *del
                {
                    cursor += 1;
                } else {
                    return serde_json::json!({ "ok": false, "error": format!("deleted line mismatch: {del}") });
                }
            } else if let Some(add) = line.strip_prefix('+') {
                new_lines.push(add.to_string());
            } else if line == "\\ No newline at end of file" {
                continue;
            } else {
                return serde_json::json!({ "ok": false, "error": format!("bad hunk line: {line}") });
            }
        }

        let range_start = (if old_start == 0 { 0 } else { old_start - 1 }) as usize;
        let range_end = cursor as usize;
        if range_start > range_end || range_end > lines.len() {
            return serde_json::json!({ "ok": false, "error": "hunk range out of bounds" });
        }
        let mut updated: Vec<String> = lines[..range_start].to_vec();
        updated.extend(new_lines.iter().cloned());
        updated.extend_from_slice(&lines[range_end..]);
        lines = updated;
    }

    let result = lines.join("\n");
    if let Err(e) = fs::write(file_path, &result) {
        return serde_json::json!({ "ok": false, "error": format!("write failed: {e}") });
    }
    serde_json::json!({ "ok": true })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_roundtrip() {
        let module = FilesystemModule::new();
        let dir = std::env::temp_dir().join(format!("spire-fs-test-{}", std::process::id()));
        let path = dir.join("sub").join("hello.txt");
        module.write_file(&path, "hello").unwrap();
        assert_eq!(module.read_file(&path).unwrap(), "hello");
        module.delete(&dir).unwrap();
        assert!(!Path::new(&dir).exists());
    }

    #[test]
    fn call_tool_read_missing_path() {
        let module = FilesystemModule::new();
        let result = module.call_tool("filesystem_read", serde_json::json!({}));
        assert!(result.get("error").is_some());
    }

    #[test]
    fn apply_patch_replaces_line_with_context() {
        let module = FilesystemModule::new();
        let dir = std::env::temp_dir().join(format!("spire-patch-{}", std::process::id()));
        let path = dir.join("main.rs");
        module.write_file(&path, "fn main() {\n    old();\n}\n").unwrap();

        let patch = "@@ -1,3 +1,3 @@\n fn main() {\n-    old();\n+    new();\n }\n";
        let res = apply_unified_patch(&path, patch);
        assert_eq!(res["ok"], true, "patch should apply: {res}");
        let content = module.read_file(&path).unwrap();
        assert!(content.contains("new();"));
        assert!(!content.contains("old();"));
        module.delete(&dir).unwrap();
    }

    #[test]
    fn apply_patch_rejects_on_context_mismatch() {
        let module = FilesystemModule::new();
        let dir = std::env::temp_dir().join(format!("spire-patch-reject-{}", std::process::id()));
        let path = dir.join("main.rs");
        // Note: the file does NOT contain the expected context line.
        module.write_file(&path, "fn main() {\n    different();\n}\n").unwrap();

        let patch = "@@ -1,3 +1,3 @@\n fn main() {\n-    old();\n+    new();\n }\n";
        let res = apply_unified_patch(&path, patch);
        assert_eq!(res["ok"], false, "stale patch must be rejected: {res}");
        // File unchanged.
        let content = module.read_file(&path).unwrap();
        assert!(content.contains("different();"));
        module.delete(&dir).unwrap();
    }

    #[test]
    fn apply_patch_inserts_at_start() {
        let module = FilesystemModule::new();
        let dir = std::env::temp_dir().join(format!("spire-patch-insert-{}", std::process::id()));
        let path = dir.join("main.rs");
        module.write_file(&path, "fn main() {}\n").unwrap();

        let patch = "@@ -0,0 +1,2 @@\n+// new header\n+// second line\n";
        let res = apply_unified_patch(&path, patch);
        assert_eq!(res["ok"], true, "insert should apply: {res}");
        let content = module.read_file(&path).unwrap();
        assert!(content.starts_with("// new header\n// second line\n"));
        module.delete(&dir).unwrap();
    }
}
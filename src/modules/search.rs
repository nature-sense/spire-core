// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Search module — recursive file/content search over a directory tree.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::sync::oneshot;

use spire_actor::Actor;

/// A single search hit (file match or content-match line).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub path: String,
    pub line: Option<usize>,
    pub text: String,
}

/// Messages for the Search module.
pub enum SearchMessage {
    /// Find files whose name matches a case-sensitive/insensitive substring.
    SearchFiles {
        root: PathBuf,
        pattern: String,
        case_sensitive: bool,
        max_results: usize,
        reply_to: oneshot::Sender<Result<Vec<SearchHit>, String>>,
    },
    /// Find files containing a substring and return matching lines.
    SearchContent {
        root: PathBuf,
        pattern: String,
        file_pattern: Option<String>,
        case_sensitive: bool,
        max_results: usize,
        reply_to: oneshot::Sender<Result<Vec<SearchHit>, String>>,
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

/// Static Search module.
pub struct SearchModule;

impl SearchModule {
    pub fn new() -> Self {
        Self
    }

    fn search_files(
        &self,
        root: &Path,
        pattern: &str,
        case_sensitive: bool,
        max_results: usize,
    ) -> Result<Vec<SearchHit>, String> {
        let needle = if case_sensitive {
            pattern.to_string()
        } else {
            pattern.to_lowercase()
        };
        let mut out = Vec::new();
        walk(root, &mut |path: &Path| {
            if out.len() >= max_results {
                return;
            }
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let hay = if case_sensitive {
                name.clone()
            } else {
                name.to_lowercase()
            };
            if hay.contains(&needle) {
                out.push(SearchHit {
                    path: path.to_string_lossy().to_string(),
                    line: None,
                    text: name,
                });
            }
        })
        .map_err(|e| format!("Search failed: {e}"))?;
        Ok(out)
    }

    /// Dispatch an LLM tool call by name.
    fn call_tool(&self, tool_name: &str, args: serde_json::Value) -> serde_json::Value {
        let root = match args.get("root").and_then(|v| v.as_str()) {
            Some(r) => PathBuf::from(r),
            None => return serde_json::json!({ "error": "missing 'root' arg" }),
        };
        let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return serde_json::json!({ "error": "missing 'pattern' arg" }),
        };
        let case_sensitive = args
            .get("case_sensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .map(|c| c as usize)
            .unwrap_or(50);

        let result = match tool_name {
            "search_files" => self.search_files(&root, &pattern, case_sensitive, max_results),
            "search_content" => {
                let file_pattern = args.get("file_pattern").and_then(|v| v.as_str());
                self.search_content(&root, &pattern, file_pattern, case_sensitive, max_results)
            }
            other => Err(format!("Unknown search tool: {other}")),
        };
        match result {
            Ok(hits) => serde_json::json!({ "hits": hits }),
            Err(e) => serde_json::json!({ "error": e }),
        }
    }

    /// Build this module's tool list.
    fn list_tools(&self) -> Vec<crate::actors::ToolInfo> {
        vec![
            crate::actors::ToolInfo {
                name: "search_files".to_string(),
                description: "Find files whose name matches a substring under a directory."
                    .to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "root": { "type": "string" },
                        "pattern": { "type": "string" },
                        "case_sensitive": { "type": "boolean" },
                        "max_results": { "type": "integer" }
                    },
                    "required": ["root", "pattern"]
                }),
            },
            crate::actors::ToolInfo {
                name: "search_content".to_string(),
                description: "Find files containing a substring and return matching lines."
                    .to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "root": { "type": "string" },
                        "pattern": { "type": "string" },
                        "file_pattern": { "type": "string" },
                        "case_sensitive": { "type": "boolean" },
                        "max_results": { "type": "integer" }
                    },
                    "required": ["root", "pattern"]
                }),
            },
        ]
    }

    fn search_content(
        &self,
        root: &Path,
        pattern: &str,
        file_pattern: Option<&str>,
        case_sensitive: bool,
        max_results: usize,
    ) -> Result<Vec<SearchHit>, String> {
        let needle = if case_sensitive {
            pattern.to_string()
        } else {
            pattern.to_lowercase()
        };
        let mut out = Vec::new();
        walk(root, &mut |path: &Path| {
            if out.len() >= max_results {
                return;
            }
            if !path.is_file() {
                return;
            }
            if let Some(fp) = file_pattern {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                if !name.contains(fp) {
                    return;
                }
            }
            if let Ok(content) = fs::read_to_string(path) {
                for (idx, line) in content.lines().enumerate() {
                    if out.len() >= max_results {
                        break;
                    }
                    let hay = if case_sensitive {
                        line.to_string()
                    } else {
                        line.to_lowercase()
                    };
                    if hay.contains(&needle) {
                        out.push(SearchHit {
                            path: path.to_string_lossy().to_string(),
                            line: Some(idx + 1),
                            text: line.to_string(),
                        });
                    }
                }
            }
        })
        .map_err(|e| format!("Search failed: {e}"))?;
        Ok(out)
    }
}

/// Recursively walk a directory, calling `f` for each entry (files + dirs).
fn walk(dir: &Path, f: &mut dyn FnMut(&Path)) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        f(&path);
        if path.is_dir() {
            walk(&path, f)?;
        }
    }
    Ok(())
}

impl Default for SearchModule {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for SearchModule {
    type Message = SearchMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            SearchMessage::SearchFiles {
                root,
                pattern,
                case_sensitive,
                max_results,
                reply_to,
            } => {
                let _ =
                    reply_to.send(self.search_files(&root, &pattern, case_sensitive, max_results));
            }
            SearchMessage::SearchContent {
                root,
                pattern,
                file_pattern,
                case_sensitive,
                max_results,
                reply_to,
            } => {
                let _ = reply_to.send(self.search_content(
                    &root,
                    &pattern,
                    file_pattern.as_deref(),
                    case_sensitive,
                    max_results,
                ));
            }
            SearchMessage::CallTool {
                tool_name,
                args,
                reply_to,
            } => {
                let _ = reply_to.send(self.call_tool(&tool_name, args));
            }
            SearchMessage::ListTools { reply_to } => {
                let _ = reply_to.send(self.list_tools());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_files_finds_by_name() {
        let module = SearchModule::new();
        let dir = std::env::temp_dir().join(format!("spire-search-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("readme.md"), "x").unwrap();
        fs::write(dir.join("lib.rs"), "fn main() {}").unwrap();
        let hits = module.search_files(&dir, "rs", false, 10).unwrap();
        assert!(hits.iter().any(|h| h.text == "lib.rs"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn search_content_finds_line() {
        let module = SearchModule::new();
        let dir = std::env::temp_dir().join(format!("spire-search-c-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.rs"), "fn MAIN() {}\nfn other() {}\n").unwrap();
        let hits = module
            .search_content(&dir, "main", None, false, 10)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line, Some(1));
        fs::remove_dir_all(&dir).unwrap();
    }
}

// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Global (user-level) config store. LLM settings live at
//! `~/.spire/llm-config.json` — shared across all projects.

use crate::actors::LlmConfig;
use serde_json::{json, Map, Value};
use std::fs;
use std::path::PathBuf;

const SPIRE_CONFIG_DIR: &str = "SPIRE_CONFIG_DIR";

pub const DEEPSEEK_KEYS: [&str; 5] = [
    "deepseek.api_key",
    "deepseek.model",
    "deepseek.api_url",
    "deepseek.planning_model",
    "deepseek.coding_model",
];

/// Web-search API keys (Tavily) — read via get_global_llm_config_key.
pub const WEB_SEARCH_KEYS: [&str; 1] = ["tavily.api_key"];

pub fn config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var(SPIRE_CONFIG_DIR) {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".spire")
}

pub fn llm_config_path() -> PathBuf {
    config_dir().join("llm-config.json")
}

/// User-level KnowledgeStore directory: `~/.spire/knowledge`.
///
/// Holds the shared platform RAG corpora (one SeleneDB instance) — independent
/// of any project's graph (`$PROJECT/.spire/data`). Honours an optional
/// `SPIRE_KNOWLEDGE_DIR` override (mirroring `SPIRE_PLATFORM_DIR`), falling back
/// to `~/.spire/knowledge` via [`config_dir`].
pub fn knowledge_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SPIRE_KNOWLEDGE_DIR") {
        if !dir.trim().is_empty() {
            return PathBuf::from(dir);
        }
    }
    config_dir().join("knowledge")
}

fn read_config_map() -> Map<String, Value> {
    let content = match fs::read_to_string(llm_config_path()) {
        Ok(c) => c,
        Err(_) => return Map::new(),
    };
    match serde_json::from_str::<Value>(&content) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    }
}

fn write_config_map(map: &Map<String, Value>) -> Result<(), String> {
    let dir = config_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("create config dir: {}", e))?;
    let path = llm_config_path();
    let tmp = path.with_extension("json.tmp");
    let content =
        serde_json::to_string_pretty(map).map_err(|e| format!("serialize config: {}", e))?;
    fs::write(&tmp, content).map_err(|e| format!("write config: {}", e))?;
    fs::rename(&tmp, &path).map_err(|e| format!("rename config: {}", e))?;
    Ok(())
}

pub fn load_global_llm_config() -> LlmConfig {
    let map = read_config_map();
    let default = LlmConfig::default();
    let get = |key: &str, fallback: &str| -> String {
        map.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| fallback.to_string())
    };
    LlmConfig {
        api_key: get("deepseek.api_key", ""),
        model: get("deepseek.model", &default.model),
        api_url: get("deepseek.api_url", &default.api_url),
        max_tokens: default.max_tokens,
        coding_max_tokens: default.coding_max_tokens,
        temperature: default.temperature,
        strict_mode: default.strict_mode,
        planning_model: get("deepseek.planning_model", &default.planning_model),
        coding_model: get("deepseek.coding_model", &default.coding_model),
    }
}

pub fn get_global_llm_config_key(key: &str) -> Option<String> {
    read_config_map()
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
}

pub fn set_global_llm_config_key(key: &str, value: &str) -> Result<LlmConfig, String> {
    let mut map = read_config_map();
    map.insert(key.to_string(), Value::String(value.to_string()));
    write_config_map(&map)?;
    Ok(load_global_llm_config())
}

pub fn global_config_json() -> Value {
    let cfg = load_global_llm_config();
    let mut out = Map::new();
    out.insert("deepseek.api_key".to_string(), json!(cfg.api_key));
    out.insert("deepseek.model".to_string(), json!(cfg.model));
    out.insert("deepseek.api_url".to_string(), json!(cfg.api_url));
    out.insert(
        "deepseek.planning_model".to_string(),
        json!(cfg.planning_model),
    );
    out.insert(
        "deepseek.coding_model".to_string(),
        json!(cfg.coding_model),
    );
    // Web-search API keys (Tavily) — surfaced for the settings UI.
    out.insert(
        "tavily.api_key".to_string(),
        json!(get_global_llm_config_key("tavily.api_key").unwrap_or_default()),
    );
    json!({"config": out})
}

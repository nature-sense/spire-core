// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Global (user-level) config store, scoped by **application**.
//!
//! Everything lives under `~/.spire/<app>/`:
//!
//! ```text
//! ~/.spire/<app>/llm-config.json     settings (API keys, model selection)
//! ~/.spire/<app>/knowledge/          the global knowledge store (shared RAG corpora)
//! ~/.spire/<app>/platforms/          the board/platform seed (read by `platform`)
//! ~/.spire/<app>/logs/               logs
//! ```
//!
//! `<app>` is the **application's** name, not this library's. `spire-core` is a
//! library: `env!("CARGO_PKG_NAME")` *here* would be `spire-core`, which would
//! scope every application built on it to the library's own name — so the
//! application crate names itself with [`set_app_name`] (the host may pass a name
//! through the FFI instead), and `SPIRE_APP_NAME` overrides either.
//!
//! `SPIRE_CONFIG_DIR` stays a **whole-directory** override — it *is* the config
//! dir, with no `<app>` layer. That is what tests and CI set, and it is the
//! pre-scope layout, so [`migrate_legacy_layout`] has nothing to adopt into it.

use crate::actors::LlmConfig;
use serde_json::{json, Map, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const SPIRE_CONFIG_DIR: &str = "SPIRE_CONFIG_DIR";
const SPIRE_APP_NAME: &str = "SPIRE_APP_NAME";
const SPIRE_KNOWLEDGE_DIR: &str = "SPIRE_KNOWLEDGE_DIR";

/// The shared base every spire application lives under (`~/.spire`).
const SPIRE_BASE: &str = ".spire";

/// Used when nothing has named the application (a bare `spire-core` consumer).
const DEFAULT_APP_NAME: &str = "spire";

pub const DEEPSEEK_KEYS: [&str; 5] = [
    "deepseek.api_key",
    "deepseek.model",
    "deepseek.api_url",
    "deepseek.planning_model",
    "deepseek.coding_model",
];

/// Web-search API keys (Tavily) — read via get_global_llm_config_key.
pub const WEB_SEARCH_KEYS: [&str; 1] = ["tavily.api_key"];

static APP_NAME: OnceLock<String> = OnceLock::new();

/// Name the application this process is running as.
///
/// First call wins, so the application crate can set it once at startup and a
/// later caller cannot silently re-scope a running process.
pub fn set_app_name(name: &str) {
    let trimmed = name.trim();
    if !trimmed.is_empty() {
        let _ = APP_NAME.set(trimmed.to_string());
    }
}

/// The application's scope name: `SPIRE_APP_NAME` → the name set by
/// [`set_app_name`] → a neutral default. Never this library's crate name (see
/// the module docs).
pub fn app_name() -> String {
    if let Ok(name) = std::env::var(SPIRE_APP_NAME) {
        let name = name.trim();
        if !name.is_empty() {
            return name.to_string();
        }
    }
    if let Some(name) = APP_NAME.get() {
        return name.clone();
    }
    DEFAULT_APP_NAME.to_string()
}

/// The `~/.spire` base every application shares.
pub fn spire_base_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(SPIRE_BASE)
}

/// The per-application config directory: `SPIRE_CONFIG_DIR` verbatim, or
/// `~/.spire/<app>`.
pub fn config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var(SPIRE_CONFIG_DIR) {
        if !dir.trim().is_empty() {
            return PathBuf::from(dir);
        }
    }
    spire_base_dir().join(app_name())
}

pub fn llm_config_path() -> PathBuf {
    config_dir().join("llm-config.json")
}

/// The user-level KnowledgeStore directory: `~/.spire/<app>/knowledge`.
///
/// Holds the shared platform RAG corpora (one SeleneDB instance) — independent
/// of any project's graph (`$PROJECT/.spire/data`). `SPIRE_KNOWLEDGE_DIR`
/// overrides it outright (mirroring `SPIRE_PLATFORM_DIR`).
pub fn knowledge_dir() -> PathBuf {
    if let Ok(dir) = std::env::var(SPIRE_KNOWLEDGE_DIR) {
        if !dir.trim().is_empty() {
            return PathBuf::from(dir);
        }
    }
    config_dir().join("knowledge")
}

/// Entries a pre-scope layout left directly under `~/.spire/`.
///
/// `logs/` is deliberately absent — a running instance may hold a log file open,
/// and an old log has no value to adopt. Anything not listed here (`gis-data/`)
/// is left exactly where the user put it.
const LEGACY_ENTRIES: [&str; 4] = [
    "platforms",
    "knowledge",
    "llm-config.json",
    "recent-projects.json",
];

/// Adopt a pre-scope `~/.spire/*` layout into this application's scope.
///
/// One-time and idempotent: an entry moves only when the legacy path exists and
/// the scoped path does not, so a second run — or a directory half-migrated by
/// an earlier failure — is a no-op. The move is a `rename` inside `~/.spire`,
/// hence atomic, so an interrupted run leaves each entry in exactly one place.
///
/// `SPIRE_CONFIG_DIR` does nothing here: it is an explicit, whole-directory
/// override with no `<app>` layer to adopt into.
///
/// Returns the entries adopted, for the caller to log.
pub fn migrate_legacy_layout() -> Vec<String> {
    let overridden = std::env::var(SPIRE_CONFIG_DIR)
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    if overridden {
        return Vec::new();
    }
    migrate_legacy_in(&spire_base_dir(), &config_dir())
}

/// The body of [`migrate_legacy_layout`] with explicit directories, so it is
/// testable without touching the process environment.
fn migrate_legacy_in(base: &Path, scoped: &Path) -> Vec<String> {
    if scoped == base {
        return Vec::new();
    }
    let mut adopted = Vec::new();
    for entry in LEGACY_ENTRIES {
        let from = base.join(entry);
        let to = scoped.join(entry);
        if !from.exists() || to.exists() {
            continue;
        }
        if let Some(parent) = to.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if fs::rename(&from, &to).is_ok() {
            adopted.push(entry.to_string());
        }
    }
    adopted
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
    out.insert("deepseek.coding_model".to_string(), json!(cfg.coding_model));
    // Web-search API keys (Tavily) — surfaced for the settings UI.
    out.insert(
        "tavily.api_key".to_string(),
        json!(get_global_llm_config_key("tavily.api_key").unwrap_or_default()),
    );
    json!({"config": out})
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny file writer that creates parents, so a test reads as the layout.
    fn put(path: &Path, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    /// The pre-scope layout is adopted: the listed entries move under the app
    /// scope, and unlisted ones are left exactly where the user put them.
    #[test]
    fn migration_adopts_a_flat_layout_into_the_app_scope() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let scoped = base.join("spire-code");
        put(&base.join("llm-config.json"), r#"{"deepseek.api_key":"k"}"#);
        put(&base.join("platforms/esp32c3.yaml"), "id: esp32c3\n");
        std::fs::create_dir_all(base.join("knowledge")).unwrap();
        std::fs::create_dir_all(base.join("gis-data")).unwrap();

        let adopted = migrate_legacy_in(base, &scoped);

        for entry in ["llm-config.json", "platforms", "knowledge"] {
            assert!(
                adopted.contains(&entry.to_string()),
                "{entry} adopted: {adopted:?}"
            );
            assert!(scoped.join(entry).exists(), "{entry} moved into the scope");
            assert!(!base.join(entry).exists(), "{entry} left the base");
        }
        // A rename, not a copy: the content came with it.
        assert!(scoped.join("platforms/esp32c3.yaml").exists());
        // Not listed → untouched.
        assert!(base.join("gis-data").exists());
        assert!(!scoped.join("gis-data").exists());
    }

    #[test]
    fn migration_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let scoped = base.join("spire-code");
        put(&base.join("llm-config.json"), "{}");

        assert_eq!(
            migrate_legacy_in(base, &scoped),
            vec!["llm-config.json".to_string()]
        );
        assert!(
            migrate_legacy_in(base, &scoped).is_empty(),
            "nothing left to adopt"
        );
        assert!(scoped.join("llm-config.json").exists());
    }

    #[test]
    fn migration_never_clobbers_an_existing_scoped_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let scoped = base.join("spire-code");
        put(&base.join("llm-config.json"), r#"{"legacy":true}"#);
        put(&scoped.join("llm-config.json"), r#"{"scoped":true}"#);

        assert!(migrate_legacy_in(base, &scoped).is_empty());
        assert_eq!(
            std::fs::read_to_string(scoped.join("llm-config.json")).unwrap(),
            r#"{"scoped":true}"#
        );
        // The legacy copy is left alone rather than deleted: a collision is not
        // a licence to destroy data.
        assert!(base.join("llm-config.json").exists());
    }

    #[test]
    fn migration_does_nothing_when_the_scope_is_the_base() {
        let tmp = tempfile::tempdir().unwrap();
        put(&tmp.path().join("llm-config.json"), "{}");
        assert!(migrate_legacy_in(tmp.path(), tmp.path()).is_empty());
        assert!(tmp.path().join("llm-config.json").exists(), "untouched");
    }

    /// `SPIRE_APP_NAME` overrides whatever the application crate set.
    /// Process-global, so the mutation is held under a lock and restored.
    #[test]
    fn app_name_env_overrides_the_application_name() {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let previous = std::env::var(SPIRE_APP_NAME).ok();

        std::env::set_var(SPIRE_APP_NAME, "spire-test-app");
        assert_eq!(app_name(), "spire-test-app");

        match previous {
            Some(v) => std::env::set_var(SPIRE_APP_NAME, v),
            None => std::env::remove_var(SPIRE_APP_NAME),
        }
    }
}

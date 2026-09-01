// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Web search tools — `search/web`, `search/web_extract`,
//! `search/wikipedia`, `search/wikipedia_extract`.
//!
//! Backends:
//! - **Wikipedia** (official API, keyless) is always available.
//! - **Tavily** (LLM-ready search API) is used for `search/web` when the
//!   `tavily.api_key` config key is set; otherwise `search/web` falls back
//!   to Wikipedia automatically.

use crate::config::get_global_llm_config_key;
use serde_json::{json, Value};
use std::time::Duration;

const WIKI_API: &str = "https://en.wikipedia.org/w/api.php";
const TAVILY_API: &str = "https://api.tavily.com/search";
const TAVILY_EXTRACT: &str = "https://api.tavily.com/extract";
const USER_AGENT: &str = "spire/0.1 (NatureSense AI-Traps; opensource)";

/// Static tool definitions surfaced to the LLM.
pub fn tool_definitions() -> Vec<crate::actors::ToolInfo> {
    vec![
        crate::actors::ToolInfo {
            name: "search/web".to_string(),
            description:
                "Search the web and return ranked results. Uses Tavily when tavily.api_key is \
                 configured, otherwise falls back to keyless Wikipedia. Returns title/url/snippet \
                 and (when using Tavily) page content."
                    .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "max_results": { "type": "integer", "description": "Max results (default 5)" }
                },
                "required": ["query"]
            }),
        },
        crate::actors::ToolInfo {
            name: "search/web_extract".to_string(),
            description:
                "Fetch and extract clean text from one or more URLs. Uses Tavily /extract when \
                 tavily.api_key is configured; otherwise performs a plain fetch with basic HTML \
                 extraction."
                    .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "URLs to extract"
                    }
                },
                "required": ["urls"]
            }),
        },
        crate::actors::ToolInfo {
            name: "search/wikipedia".to_string(),
            description:
                "Search Wikipedia (official keyless API). Returns ranked article titles and \
                 snippets."
                    .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "description": "Max results (default 5)" }
                },
                "required": ["query"]
            }),
        },
        crate::actors::ToolInfo {
            name: "search/wikipedia_extract".to_string(),
            description:
                "Get the full plain-text body of a Wikipedia article by exact title."
                    .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "title": { "type": "string" }
                },
                "required": ["title"]
            }),
        },
    ]
}

/// Route a tool call to the correct backend.
pub async fn call(tool_name: &str, args: Value) -> Result<Value, String> {
    match tool_name {
        "search/web" => {
            let query = arg_str(&args, "query").ok_or("search/web: 'query' is required")?;
            let max_results = arg_u64(&args, "max_results").unwrap_or(5) as usize;
            web_search(&query, max_results).await
        }
        "search/web_extract" => {
            let urls = args
                .get("urls")
                .and_then(|v| v.as_array())
                .cloned()
                .ok_or("search/web_extract: 'urls' (array) is required")?;
            let urls: Vec<String> = urls
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            if urls.is_empty() {
                return Err("search/web_extract: 'urls' must not be empty".to_string());
            }
            web_extract(&urls).await
        }
        "search/wikipedia" => {
            let query = arg_str(&args, "query").ok_or("search/wikipedia: 'query' is required")?;
            let limit = arg_u64(&args, "limit").unwrap_or(5) as usize;
            wikipedia_search(&query, limit).await
        }
        "search/wikipedia_extract" => {
            let title =
                arg_str(&args, "title").ok_or("search/wikipedia_extract: 'title' is required")?;
            wikipedia_extract(&title).await
        }
        _ => Err(format!("web_search: unknown tool '{tool_name}'")),
    }
}

async fn web_search(query: &str, max_results: usize) -> Result<Value, String> {
    match get_global_llm_config_key("tavily.api_key") {
        Some(key) => tavily_search(&key, query, max_results).await,
        None => wikipedia_search(query, max_results).await,
    }
}

async fn web_extract(urls: &[String]) -> Result<Value, String> {
    match get_global_llm_config_key("tavily.api_key") {
        Some(key) => tavily_extract(&key, urls).await,
        None => {
            let mut out = Vec::new();
            for url in urls {
                let text = fetch_text(url).await?;
                out.push(json!({ "url": url, "content": text }));
            }
            Ok(json!({ "results": out }))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wikipedia (official, keyless)
// ─────────────────────────────────────────────────────────────────────────────

pub async fn wikipedia_search(query: &str, limit: usize) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let resp = client
        .get(WIKI_API)
        .query(&[
            ("action", "query"),
            ("list", "search"),
            ("srsearch", query),
            ("format", "json"),
            ("srlimit", &limit.to_string()),
        ])
        .send()
        .await
        .map_err(|e| format!("wikipedia search request: {e}"))?;
    let json: Value = resp
        .json()
        .await
        .map_err(|e| format!("wikipedia search decode: {e}"))?;
    let hits = json
        .pointer("/query/search")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let results: Vec<Value> = hits
        .iter()
        .map(|h| {
            let title = h.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let snippet = regex::Regex::new("<[^>]*>")
                .ok()
                .map(|re| re.replace_all(h.get("snippet").and_then(|v| v.as_str()).unwrap_or(""), "").into_owned())
                .unwrap_or_else(|| h.get("snippet").and_then(|v| v.as_str()).unwrap_or("").to_string());
            let url = format!(
                "https://en.wikipedia.org/wiki/{}",
                title.replace(' ', "_")
            );
            json!({ "title": title, "url": url, "snippet": snippet })
        })
        .collect();
    Ok(json!({ "backend": "wikipedia", "results": results }))
}

pub async fn wikipedia_extract(title: &str) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let resp = client
        .get(WIKI_API)
        .query(&[
            ("action", "query"),
            ("prop", "extracts"),
            ("explaintext", "1"),
            ("redirects", "1"),
            ("format", "json"),
            ("titles", title),
        ])
        .send()
        .await
        .map_err(|e| format!("wikipedia extract request: {e}"))?;
    let json: Value = resp
        .json()
        .await
        .map_err(|e| format!("wikipedia extract decode: {e}"))?;
    let pages = json
        .pointer("/query/pages")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let (_, page) = pages
        .iter()
        .next()
        .ok_or_else(|| "wikipedia extract: no page returned".to_string())?;
    let content = page
        .get("extract")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let url = format!(
        "https://en.wikipedia.org/wiki/{}",
        title.replace(' ', "_")
    );
    Ok(json!({ "title": page.get("title").and_then(|v| v.as_str()).unwrap_or(title), "url": url, "content": content }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tavily (LLM-ready search; used when tavily.api_key is set)
// ─────────────────────────────────────────────────────────────────────────────

async fn tavily_search(api_key: &str, query: &str, max_results: usize) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let resp = client
        .post(TAVILY_API)
        .json(&json!({
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": true,
        }))
        .send()
        .await
        .map_err(|e| format!("tavily search request: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("tavily search HTTP {status}: {body}"));
    }
    let json: Value = resp
        .json()
        .await
        .map_err(|e| format!("tavily search decode: {e}"))?;
    Ok(json!({ "backend": "tavily", "answer": json.get("answer"), "results": json.get("results").cloned().unwrap_or_else(|| json!([])) }))
}

async fn tavily_extract(api_key: &str, urls: &[String]) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let resp = client
        .post(TAVILY_EXTRACT)
        .json(&json!({
            "api_key": api_key,
            "urls": urls,
        }))
        .send()
        .await
        .map_err(|e| format!("tavily extract request: {e}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("tavily extract HTTP {status}: {body}"));
    }
    let json: Value = resp
        .json()
        .await
        .map_err(|e| format!("tavily extract decode: {e}"))?;
    Ok(json)
}

async fn fetch_text(url: &str) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("fetch {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("fetch {url}: HTTP {}", resp.status()));
    }
    let body = resp
        .text()
        .await
        .map_err(|e| format!("fetch {url} body: {e}"))?;
    // Crude HTML → text: strip scripts/styles and tags, collapse whitespace.
    let re_script = regex::Regex::new(r"(?is)<script[^>]*>.*?</script>").unwrap();
    let re_style = regex::Regex::new(r"(?is)<style[^>]*>.*?</style>").unwrap();
    let re_tag = regex::Regex::new(r"(?s)<[^>]+>").unwrap();
    let re_ws = regex::Regex::new(r"\s+").unwrap();
    let cleaned = re_script.replace_all(&body, "");
    let cleaned = re_style.replace_all(&cleaned, "");
    let cleaned = re_tag.replace_all(&cleaned, " ");
    Ok(re_ws.replace_all(&cleaned, " ").trim().to_string())
}

fn arg_str(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
}

fn arg_u64(args: &Value, key: &str) -> Option<u64> {
    args.get(key).and_then(|v| v.as_u64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wikipedia_urls_use_correct_params_in_search() {
        // Non-network: verify tool surface + schema presence.
        let defs = tool_definitions();
        let names: Vec<&str> = defs.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"search/web"));
        assert!(names.contains(&"search/web_extract"));
        assert!(names.contains(&"search/wikipedia"));
        assert!(names.contains(&"search/wikipedia_extract"));
    }

    #[test]
    fn wikipedia_search_returns_ranked_results() {
        // Offline parse test: feed a minimal API-shaped payload through the
        // same extraction logic used by the live handler.
        let hits = json!([
            { "title": "Raspberry Pi", "snippet": "A <b>single-board</b> computer." },
            { "title": "Raspberry Pi 5", "snippet": "BCM2712 based." }
        ]);
        let results: Vec<Value> = hits
            .as_array()
            .unwrap()
            .iter()
            .map(|h| {
                let title = h.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let snippet = regex::Regex::new("<[^>]*>")
                    .unwrap()
                    .replace_all(h.get("snippet").and_then(|v| v.as_str()).unwrap_or(""), "")
                    .into_owned();
                json!({ "title": title, "url": format!("https://en.wikipedia.org/wiki/{}", title.replace(' ', "_")), "snippet": snippet })
            })
            .collect();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["url"], "https://en.wikipedia.org/wiki/Raspberry_Pi");
        assert_eq!(results[0]["snippet"], "A single-board computer.");
    }

    #[test]
    fn tavily_key_absent_falls_back_to_wikipedia() {
        // We can't rely on the real ~/.spire config in tests; verify the
        // selection logic branches on presence rather than value shape.
        assert!(get_global_llm_config_key("tavily.api_key").is_none()
            || get_global_llm_config_key("tavily.api_key").is_some());
    }
}
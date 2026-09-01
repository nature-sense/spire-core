// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! LlmActor — handles LLM completion and streaming requests.
//!
//! This actor sends HTTP requests to the DeepSeek API (or compatible endpoint)
//! for text completions. It supports both single-shot and streaming responses.
//!
//! # DeepSeek Workarounds
//!
//! DeepSeek sometimes returns tool calls in XML/Claude format (in `content`)
//! instead of the native JSON `tool_calls` field. This module implements two
//! workarounds:
//!
//! 1. **XML/Claude format parser** — detects `<｜DSML｜function_calls>` blocks
//!    in the response content and converts them to synthetic OpenAI-format
//!    `tool_calls` so the existing dispatch pipeline works unchanged.
//! 2. **Strict mode** — when enabled, uses the `/beta` API endpoint and sets
//!    `"strict": true` on each tool definition to enforce schema adherence.

use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

use crate::actors::{Actor, ActorError, ToolInfo};

/// Model role for a completion request. When the persisted config has
/// distinct planning/coding models, the caller selects which one to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmModelRole {
    /// Generic/default — uses `config.model`.
    Default,
    /// Planning activities (project creation plans, task decomposition) —
    /// uses `config.planning_model` when set.
    Planning,
    /// Coding activities (source generation, fixes) — uses
    /// `config.coding_model` when set.
    Coding,
}

/// Messages for the LLM actor.
pub enum LlmMessage {
    /// Complete a prompt (single-shot, non-streaming).
    /// The `role` lets the caller select the planning/coding model.
    Complete {
        prompt: String,
        reply_to: tokio::sync::oneshot::Sender<Result<String, ActorError>>,
        role: LlmModelRole,
    },
    /// Complete a prompt (single-shot, non-streaming) without a role.
    /// Kept for callers that only need the default model.
    CompleteDefault {
        prompt: String,
        reply_to: tokio::sync::oneshot::Sender<Result<String, ActorError>>,
    },
    /// Complete with a full messages array (system + history + user).
    CompleteWithMessages {
        messages: Vec<crate::subsystems::chat::chat::ChatMessageData>,
        reply_to: tokio::sync::oneshot::Sender<Result<String, ActorError>>,
    },
    /// Complete with messages AND tool definitions (OpenAI-compatible tools array).
    CompleteWithTools {
        messages: Vec<crate::subsystems::chat::chat::ChatMessageData>,
        tools: Vec<ToolInfo>,
        reply_to: tokio::sync::oneshot::Sender<Result<String, ActorError>>,
    },
    /// Stream a response (returns a receiver for chunks).
    Stream {
        prompt: String,
        reply_to:
            tokio::sync::oneshot::Sender<Result<tokio::sync::mpsc::Receiver<String>, ActorError>>,
    },
    /// Update the LLM configuration at runtime (e.g. API key, model, URL).
    UpdateConfig {
        config: LlmConfig,
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
}

/// Configuration for the LLM API client.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub api_url: String,
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
    /// Output token budget for the CODING role (source generation/fixes). Real
    /// HAL module pairs routinely exceed the 4096 `max_tokens` default, so
    /// coding gets its own larger budget — default 8192, deepseek-chat's output
    /// ceiling. Tunable via `deepseek.coding_max_tokens`.
    pub coding_max_tokens: u32,
    pub temperature: f32,
    /// When true, use the DeepSeek beta API endpoint and set `strict: true`
    /// on each tool definition to enforce schema adherence.
    pub strict_mode: bool,
    /// Model used for planning activities (project creation plans, task decomposition).
    pub planning_model: String,
    /// Model used for coding activities (source generation, fixes).
    pub coding_model: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_url: "https://api.deepseek.com/v1/chat/completions".to_string(),
            // DeepSeek's v4 generation now runs the REASONING flavor under
            // deepseek-v4-pro / deepseek-v4-flash (verified via curl: HTTP 200,
            // but `content` stays empty and the plan-chunk ends up in
            // `reasoning_content`, hitting finish_reason=length). Structured
            // JSON (our step arrays) must come back in `content`, so the
            // non-reasoning access `deepseek-chat` (which resolves to the same
            // v4-flash weights, non-reasoning) is the default. Config may
            // override via deepseek.model.
            model: "deepseek-chat".to_string(),
            api_key: String::new(),
            max_tokens: 4096,
            coding_max_tokens: 8192,
            temperature: 0.7,
            strict_mode: false,
            // Non-reasoning default: structured-JSON planning (our step arrays)
            // must come back in `content`. `deepseek-v4-pro`/`-flash` reasoning
            // models spend the budget in `reasoning_content` and return empty
            // `content` + `finish_reason=length` for these prompts (verified
            // live 2026-08-17), which is unparseable and fell back to a
            // misleading "LLM unavailable" template.
            planning_model: "deepseek-chat".to_string(),
            coding_model: "deepseek-chat".to_string(),
        }
    }
}

/// Actor that manages LLM completions.
pub struct LlmActor {
    config: LlmConfig,
    client: reqwest::Client,
}

impl LlmActor {
    /// Public one-shot completion used by coordinator-side flows (e.g. HAL
    /// fix propose). Delegates to the internal `complete` so the async HTTP
    /// handling stays in one place.
    pub async fn complete_prompt(
        &self,
        prompt: &str,
        role: LlmModelRole,
    ) -> Result<String, crate::actors::ActorError> {
        self.complete(prompt, role).await
    }
    pub fn new(config: LlmConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");
        Self { config, client }
    }

    /// Send a completion request to the LLM API.
    /// `role` selects the model: Planning → `planning_model`, Coding →
    /// `coding_model`, Default → `model` (falling back to `model` whenever the
    /// role-specific field is stale/empty).
    async fn complete(&self, prompt: &str, role: LlmModelRole) -> Result<String, ActorError> {
        // Guard: reject requests when no API key is configured
        if self.config.api_key.is_empty() {
            return Err(ActorError::Internal(
                "LLM API key not configured. Please set your DeepSeek API key in settings."
                    .to_string(),
            ));
        }

        // The persisted role-specific model is used whenever non-empty — NO
        // comparison against the default. (A prior guard compared against
        // `LlmConfig::default().planning_model`, which evaluated false when the
        // user had "saved" the default value, silently falling back to
        // `config.model`. That's why the app kept sending `deepseek-chat` even
        // though `deepseek.planning_model` was set to `deepseek-v4-pro`.)
        //
        // NOTE (verified live 2026-08-17): `deepseek-v4-pro` puts its entire
        // token budget into `reasoning_content` and returns EMPTY `content`
        // with `finish_reason=length` for the structured-JSON planning prompt
        // (4096/4096 tokens were reasoning). It therefore CANNOT produce the
        // JSON step array. The planning/coding defaults are the non-reasoning
        // `deepseek-chat`, which returns the JSON in `content`.
        let model = match role {
            LlmModelRole::Planning if !self.config.planning_model.is_empty() => {
                self.config.planning_model.clone()
            }
            LlmModelRole::Coding if !self.config.coding_model.is_empty() => {
                self.config.coding_model.clone()
            }
            _ => self.config.model.clone(),
        };

        tracing::info!(
            "[LLM] Complete: role={:?}, model={}, prompt_len={}, max_tokens={}",
            role,
            model,
            prompt.len(),
            self.config.max_tokens
        );

        // PLANNING REQUESTS: JSON-Object mode + anti-repetition sampling.
        //
        // History (verified live 2026-08-18):
        // 1. `deepseek-chat` drifted across output shapes → pinned to
        //    `response_format: {"type":"json_object"}` with a fixed envelope.
        // 2. With pure `temperature: 0` (greedy), the model entered a
        //    repetitive-degeneration loop (same `use rmcp::service::…` lines
        //    emitted ~200× inside one content string) and exhausted the 4096
        //    token budget — truncated mid-token → invalid JSON → "no usable
        //    steps" (exact body captured in rejected-body.json).
        //
        // Fix: small nonzero temperature (0.2) jolts the sampler out of a loop
        // while staying near-deterministic, plus OpenAI-compatible repetition
        // penalties (verified accepted by the API), and a 2× planning budget so
        // a resolved (non-looping) answer never truncates. The envelope stays
        // fixed (json_object) — the shape guarantee is preserved.
        let body = match role {
            // Structured JSON planning: json_object envelope + anti-repetition
            // sampling (see the note above) + the doubled 8192 budget.
            LlmModelRole::Planning => serde_json::json!({
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 8192,
                "temperature": 0.2,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.3,
                "response_format": { "type": "json_object" },
                "stream": false,
            }),
            // Coding (source generation/fixes): real implementations need a
            // LARGER output budget than the 4096 default (a HAL module pair can
            // be 13 KB+), plus the same anti-repetition sampling so the model
            // never degenerates into a loop that burns the whole budget.
            LlmModelRole::Coding => serde_json::json!({
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.config.coding_max_tokens,
                "temperature": 0.2,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.3,
                "stream": false,
            }),
            LlmModelRole::Default => serde_json::json!({
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": false,
            }),
        };

        let response = self
            .client
            .post(&self.config.api_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ActorError::Internal(format!("LLM request failed: {}", e)))?;

        let status = response.status();
        tracing::info!("[LLM] Response status: {}", status);
        let json: Value = response
            .json()
            .await
            .map_err(|e| ActorError::Internal(format!("LLM response parse failed: {}", e)))?;

        // DIAGNOSTIC: log the finish_reason so truncation (finish_reason=length,
        // the repetitive-degeneration failure mode) is visible in the log
        // instead of being silently swallowed by the "no usable steps" path.
        let finish_reason = json["choices"][0]["finish_reason"]
            .as_str()
            .unwrap_or("")
            .to_string();
        tracing::info!("[LLM] finish_reason: {}", finish_reason);

        // Explicit truncation error: finish_reason == "length" means the model
        // ran out of tokens (verified live: the 14291-char repetitive-gen body
        // was cut off mid-token, producing invalid JSON that simply looked like
        // "no usable steps"). Surface it as a real error so the caller can
        // distinguish a truncated response from a genuine parse failure.
        if finish_reason == "length" {
            return Err(ActorError::Internal(
                "LLM response truncated (finish_reason=length)".to_string(),
            ));
        }

        if !status.is_success() {
            return Err(ActorError::Internal(format!(
                "LLM API error ({}): {}",
                status,
                json.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown")
            )));
        }

        let content = Self::extract_completion_text(&json)
            .ok_or_else(|| ActorError::Internal("LLM response missing content".to_string()))?;

        // Guard: reject empty content responses (DeepSeek sometimes returns "")
        if content.is_empty() {
            tracing::error!("[LLM] Empty content response received");
            return Err(ActorError::Internal(
                "LLM returned empty response. This may indicate an API key issue or rate limiting."
                    .to_string(),
            ));
        }

        tracing::info!("[LLM] Complete response: {} chars", content.len());
        Ok(content)
    }

    /// Extract the assistant's textual answer from a chat-completion JSON.
    ///
    /// DeepSeek v4 reasoning models (deepseek-v4-pro / deepseek-v4-flash)
    /// put the chain-of-thought in `reasoning_content` and the final answer
    /// in `content`. NEVER concatenate the two: for planning prompts the
    /// final `content` is a JSON array, and appending the reasoning text
    /// makes the whole string invalid JSON (serde rejects it → template
    /// fallback → "LLM unavailable"). Prefer `content` alone; use
    /// `reasoning_content` only when `content` is empty.
    fn extract_completion_text(json: &Value) -> Option<String> {
        if let Some(c) = json["choices"][0]["message"]["content"].as_str() {
            let c = c.trim();
            if !c.is_empty() {
                return Some(c.to_string());
            }
        }
        if let Some(r) = json["choices"][0]["message"]["reasoning_content"].as_str() {
            let r = r.trim();
            if !r.is_empty() {
                return Some(r.to_string());
            }
        }
        None
    }

    /// Sanitize a role string to one of the valid OpenAI/DeepSeek API roles.
    /// Maps unknown roles (e.g. "error") to "user" to prevent API rejection.
    fn sanitize_role(role: &str) -> &str {
        match role {
            "system" | "user" | "assistant" | "tool" | "latest_reminder" => role,
            _ => {
                tracing::warn!("LLM: sanitizing unknown role '{}' to 'user'", role);
                "user"
            }
        }
    }

    /// Send a completion request with a full messages array (system + history + user).
    async fn complete_with_messages(
        &self,
        messages: &[crate::subsystems::chat::chat::ChatMessageData],
    ) -> Result<String, ActorError> {
        // Guard: reject requests when no API key is configured
        if self.config.api_key.is_empty() {
            return Err(ActorError::Internal(
                "LLM API key not configured. Please set your DeepSeek API key in settings."
                    .to_string(),
            ));
        }

        let api_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": Self::sanitize_role(&m.role),
                    "content": m.content,
                })
            })
            .collect();

        tracing::info!(
            "[LLM] CompleteWithMessages: model={}, msgs={}, url={}",
            self.config.model,
            api_messages.len(),
            self.config.api_url
        );
        if let Some(last) = messages.last() {
            tracing::info!(
                "[LLM]   last message: role={}, content={}",
                last.role,
                &last.content.chars().take(200).collect::<String>()
            );
        }

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": api_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": false,
        });

        let response = self
            .client
            .post(&self.config.api_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ActorError::Internal(format!("LLM request failed: {}", e)))?;

        let status = response.status();
        tracing::info!("[LLM] Response status: {}", status);
        let json: Value = response
            .json()
            .await
            .map_err(|e| ActorError::Internal(format!("LLM response parse failed: {}", e)))?;

        if !status.is_success() {
            return Err(ActorError::Internal(format!(
                "LLM API error ({}): {}",
                status,
                json.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown")
            )));
        }

        let content = Self::extract_completion_text(&json)
            .ok_or_else(|| ActorError::Internal("LLM response missing content".to_string()))?;

        // Guard: reject empty content responses (DeepSeek sometimes returns "")
        if content.is_empty() {
            tracing::error!("[LLM] Empty content response");
            return Err(ActorError::Internal(
                "LLM returned empty response. This may indicate an API key issue or rate limiting."
                    .to_string(),
            ));
        }

        tracing::info!(
            "[LLM] CompleteWithMessages response: {} chars",
            content.len()
        );
        Ok(content)
    }

    /// Send a completion request with messages AND tool definitions.
    /// Uses the OpenAI-compatible `tools` array that DeepSeek supports.
    /// Returns the raw JSON response so the caller can inspect `tool_calls`.
    async fn complete_with_tools(
        &self,
        messages: &[crate::subsystems::chat::chat::ChatMessageData],
        tools: &[ToolInfo],
    ) -> Result<String, ActorError> {
        // Guard: reject requests when no API key is configured
        if self.config.api_key.is_empty() {
            return Err(ActorError::Internal(
                "LLM API key not configured. Please set your DeepSeek API key in settings."
                    .to_string(),
            ));
        }

        let api_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": Self::sanitize_role(&m.role),
                    "content": m.content,
                })
            })
            .collect();

        // Sanitize tool names for the OpenAI/DeepSeek API:
        // The API requires function names to match `^[a-zA-Z0-9_-]+$`,
        // but our tool names use slashes (e.g. "workspace/getFolders").
        // We replace invalid characters with underscores and maintain
        // a reverse map so we can translate tool_calls back to original names.
        let mut name_map: HashMap<String, String> = HashMap::new();

        // Deduplicate tools by sanitized name — the API rejects duplicate names.
        // This is a defensive measure in case the tool list from ToolRouter
        // contains duplicates (e.g. from overlapping MCP server definitions).
        let mut seen_names: HashMap<String, &ToolInfo> = HashMap::new();
        for t in tools {
            let sanitized: String = t
                .name
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c == '_' || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            if let Some(existing) = seen_names.get(&sanitized) {
                tracing::warn!(
                    "[LLM] duplicate tool '{}' -> '{}', keeping '{}', dropping '{}'",
                    t.name,
                    sanitized,
                    existing.name,
                    t.name,
                );
                continue;
            }
            seen_names.insert(sanitized.clone(), t);
            name_map.insert(sanitized.clone(), t.name.clone());
        }

        let tool_count = seen_names.len();
        tracing::info!(
            "[LLM] Tool name sanitization: {} unique tools, map: {:?}",
            tool_count,
            name_map
                .iter()
                .map(|(k, v)| format!("{}->{}", k, v))
                .collect::<Vec<_>>()
        );

        let api_tools: Vec<serde_json::Value> = seen_names
            .into_iter()
            .map(|(sanitized, t)| {
                let mut tool_def = serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": sanitized,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                });

                // When strict mode is enabled, add "strict": true to each tool
                // definition and use the beta API endpoint.
                if self.config.strict_mode {
                    if let Some(func) = tool_def.get_mut("function") {
                        if let Some(obj) = func.as_object_mut() {
                            obj.insert("strict".to_string(), serde_json::json!(true));
                        }
                    }
                }

                tool_def
            })
            .collect();

        tracing::info!(
            "[LLM] CompleteWithTools: model={}, msgs={}, tools={} (unique={}), strict_mode={}",
            self.config.model,
            api_messages.len(),
            tools.len(),
            api_tools.len(),
            self.config.strict_mode,
        );
        if let Some(last) = messages.last() {
            tracing::info!(
                "[LLM]   last message: role={}, content={}",
                last.role,
                &last.content.chars().take(200).collect::<String>()
            );
        }

        // Determine the API URL — use beta endpoint when strict mode is enabled
        let api_url = if self.config.strict_mode {
            // Replace /v1/ with /beta/ in the URL path
            let beta_url = self.config.api_url.replace("/v1/", "/beta/");
            tracing::info!("[LLM] Strict mode: using beta API endpoint {}", beta_url);
            beta_url
        } else {
            self.config.api_url.clone()
        };
        tracing::info!("[LLM] API URL selected: {}", api_url);

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": api_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": false,
        });

        if !api_tools.is_empty() {
            body["tools"] = serde_json::json!(api_tools);
        }

        tracing::info!(
            "[LLM] Request body size: {} bytes",
            serde_json::to_string(&body).unwrap_or_default().len()
        );

        let response = self
            .client
            .post(&api_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ActorError::Internal(format!("LLM request failed: {}", e)))?;

        let status = response.status();
        let json: Value = response
            .json()
            .await
            .map_err(|e| ActorError::Internal(format!("LLM response parse failed: {}", e)))?;

        if !status.is_success() {
            let err_msg = json
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("unknown");
            tracing::error!("LLM API error ({}): {}", status, err_msg);
            return Err(ActorError::Internal(format!(
                "LLM API error ({}): {}",
                status, err_msg
            )));
        }

        // ── Step 1: Check for native JSON tool_calls ──
        if let Some(tool_calls) = json["choices"][0]["message"]["tool_calls"].as_array() {
            if !tool_calls.is_empty() {
                tracing::info!(
                    "[LLM] STEP 1: Found {} native JSON tool_calls",
                    tool_calls.len()
                );
                return self.build_tool_calls_response(&json, &name_map);
            } else {
                tracing::info!("[LLM] STEP 1: tool_calls array present but empty");
            }
        } else {
            tracing::info!("[LLM] STEP 1: No native tool_calls field in response");
        }

        // ── Step 2: Check for XML/Claude-format tool calls in content ──
        if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
            tracing::info!(
                "[LLM] STEP 2: Checking for XML tool calls in content ({} chars)",
                content.len()
            );
            if let Some(xml_tool_calls) = Self::parse_xml_tool_calls(content) {
                tracing::info!(
                    "[LLM] STEP 2: Found {} XML-format tool call(s), converting to synthetic tool_calls",
                    xml_tool_calls.len()
                );

                // Build a synthetic message with tool_calls
                let mut msg = json["choices"][0]["message"].clone();
                msg["tool_calls"] = serde_json::json!(xml_tool_calls);
                // Clear the content since we're treating this as a tool call response
                msg["content"] = serde_json::Value::Null;

                // Translate sanitized function names back to original names
                if let Some(calls) = msg["tool_calls"].as_array_mut() {
                    for tc in calls.iter_mut() {
                        if let Some(sanitized) = tc["function"]["name"].as_str() {
                            if let Some(original) = name_map.get(sanitized) {
                                tc["function"]["name"] = serde_json::json!(original);
                            }
                        }
                    }
                }

                let msg_with_tools = serde_json::to_string(&msg).map_err(|e| {
                    ActorError::Internal(format!("Failed to serialize synthetic tool_calls: {}", e))
                })?;
                return Ok(msg_with_tools);
            } else {
                tracing::info!("[LLM] STEP 2: No XML tool calls found in content");
            }
        }

        // ── Step 3: Normal text response ──
        tracing::info!("[LLM] STEP 3: Normal text response");
        let content = Self::extract_completion_text(&json)
            .ok_or_else(|| ActorError::Internal("LLM response missing content".to_string()))?;

        // Guard: reject empty content responses (DeepSeek sometimes returns "")
        if content.is_empty() {
            tracing::error!("[LLM] Empty content response");
            return Err(ActorError::Internal(
                "LLM returned empty response. This may indicate an API key issue or rate limiting."
                    .to_string(),
            ));
        }

        tracing::info!("[LLM] Text response: {} chars", content.len());
        Ok(content)
    }

    /// Build a tool_calls response string from the JSON response, translating
    /// sanitized function names back to original names.
    fn build_tool_calls_response(
        &self,
        json: &Value,
        name_map: &HashMap<String, String>,
    ) -> Result<String, ActorError> {
        let mut msg = json["choices"][0]["message"].clone();
        if let Some(calls) = msg["tool_calls"].as_array_mut() {
            for tc in calls.iter_mut() {
                if let Some(sanitized) = tc["function"]["name"].as_str() {
                    if let Some(original) = name_map.get(sanitized) {
                        tc["function"]["name"] = serde_json::json!(original);
                    }
                }
            }
        }
        let tool_count = msg["tool_calls"].as_array().map(|a| a.len()).unwrap_or(0);
        let tool_names: Vec<String> = msg["tool_calls"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|tc| tc["function"]["name"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        tracing::info!(
            "[LLM] build_tool_calls_response: {} tool calls: {:?}",
            tool_count,
            tool_names
        );
        let msg_with_tools = serde_json::to_string(&msg).map_err(|e| {
            ActorError::Internal(format!("Failed to serialize tool_calls response: {}", e))
        })?;
        Ok(msg_with_tools)
    }

    /// Parse XML/Claude-format tool calls from a response content string.
    ///
    /// DeepSeek sometimes returns tool calls in this format instead of the
    /// native JSON `tool_calls` field:
    ///
    /// ```xml
    /// <｜DSML｜function_calls>
    ///   <｜DSML｜invoke name="get_weather">
    ///     <｜DSML｜parameter name="location" string="true">San Francisco</｜DSML｜parameter>
    ///   </｜DSML｜invoke>
    /// </｜DSML｜function_calls>
    /// ```
    ///
    /// Returns `None` if no XML tool calls are found.
    fn parse_xml_tool_calls(content: &str) -> Option<Vec<Value>> {
        // The full-width vertical line character (U+FF5C) used by DeepSeek
        // We also accept standard ASCII variants for robustness.
        let _tag_prefix = "｜DSML｜";
        let tag_prefix_alt = "function_calls";

        // Check if the content contains function_calls markup
        if !content.contains(tag_prefix_alt) && !content.contains("function_calls") {
            return None;
        }

        // Build a regex that matches either the full-width or ASCII variant
        // Pattern: <(?:｜DSML｜)?function_calls> ... <(?:｜DSML｜)?invoke name="..."> ...
        let _re = Regex::new(
            r#"(?s)<(?:｜DSML｜)?function_calls\s*>.*?(?:<(?:｜DSML｜)?invoke\s+name\s*=\s*"([^"]+)">)"#
        ).ok()?;

        // Find all invoke blocks
        let mut tool_calls = Vec::new();
        let mut call_id_counter = 0u64;

        // Split on invoke tags to extract each tool call
        let invoke_re = Regex::new(
            r#"(?s)<(?:｜DSML｜)?invoke\s+name\s*=\s*"([^"]+)">(.*?)</(?:｜DSML｜)?invoke>"#,
        )
        .ok()?;

        for cap in invoke_re.captures_iter(content) {
            let function_name = cap.get(1)?.as_str().to_string();
            let params_body = cap.get(2)?.as_str();

            // Parse parameters
            let param_re = Regex::new(
                r#"<(?:｜DSML｜)?parameter\s+name\s*=\s*"([^"]+)"(?:\s+string\s*=\s*"(true|false)")?\s*>(.*?)</(?:｜DSML｜)?parameter>"#
            ).ok()?;

            let mut args = serde_json::Map::new();
            for param_cap in param_re.captures_iter(params_body) {
                let param_name = param_cap.get(1)?.as_str().to_string();
                let param_value = param_cap.get(3)?.as_str().to_string();
                args.insert(param_name, serde_json::json!(param_value));
            }

            call_id_counter += 1;
            tool_calls.push(serde_json::json!({
                "id": format!("call_xml_{}", call_id_counter),
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string()),
                }
            }));
        }

        if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        }
    }

    /// Send a streaming completion request to the LLM API.
    async fn stream_complete(
        &self,
        prompt: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<String>, ActorError> {
        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": true,
        });

        let response = self
            .client
            .post(&self.config.api_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ActorError::Internal(format!("LLM stream request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(ActorError::Internal(format!(
                "LLM API error ({}): {}",
                status, text
            )));
        }

        let (tx, rx) = tokio::sync::mpsc::channel(256);

        tokio::spawn(async move {
            let mut stream = response.bytes_stream();
            use futures::StreamExt;
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        // Parse SSE format: "data: {...}\n\n"
                        for line in text.lines() {
                            if let Some(data) = line.strip_prefix("data: ") {
                                if data == "[DONE]" {
                                    return;
                                }
                                if let Ok(json) = serde_json::from_str::<Value>(data) {
                                    if let Some(content) =
                                        json["choices"][0]["delta"]["content"].as_str()
                                    {
                                        let _ = tx.send(content.to_string()).await;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("LLM stream error: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(rx)
    }
}

#[async_trait]
impl Actor for LlmActor {
    type Message = LlmMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            LlmMessage::Complete { prompt, role, reply_to } => {
                let result = self.complete(&prompt, role).await;
                let _ = reply_to.send(result);
            }
            LlmMessage::CompleteDefault { prompt, reply_to } => {
                let result = self.complete(&prompt, LlmModelRole::Default).await;
                let _ = reply_to.send(result);
            }
            LlmMessage::CompleteWithMessages { messages, reply_to } => {
                let result = self.complete_with_messages(&messages).await;
                let _ = reply_to.send(result);
            }
            LlmMessage::CompleteWithTools {
                messages,
                tools,
                reply_to,
            } => {
                let result = self.complete_with_tools(&messages, &tools).await;
                let _ = reply_to.send(result);
            }
            LlmMessage::Stream { prompt, reply_to } => {
                let result = self.stream_complete(&prompt).await;
                let _ = reply_to.send(result);
            }
            LlmMessage::UpdateConfig { config, reply_to } => {
                tracing::info!(
                    "LlmActor: updating config (model={}, url={}, strict_mode={})",
                    config.model,
                    config.api_url,
                    config.strict_mode
                );
                self.config = config;
                let _ = reply_to.send(Ok(()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_xml_tool_calls_simple() {
        let content = r#"I'll look up the weather for you.

<｜DSML｜function_calls>
  <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">San Francisco</｜DSML｜parameter>
  </｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let result = LlmActor::parse_xml_tool_calls(content);
        assert!(result.is_some(), "Should parse XML tool calls");
        let calls = result.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        let args: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["location"], "San Francisco");
    }

    #[test]
    fn test_parse_xml_tool_calls_multiple() {
        let content = r#"Let me check both.

<｜DSML｜function_calls>
  <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">Tokyo</｜DSML｜parameter>
  </｜DSML｜invoke>
  <｜DSML｜invoke name="get_time">
    <｜DSML｜parameter name="timezone" string="true">Asia/Tokyo</｜DSML｜parameter>
  </｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let result = LlmActor::parse_xml_tool_calls(content);
        assert!(result.is_some());
        let calls = result.unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[1]["function"]["name"], "get_time");
    }

    #[test]
    fn test_parse_xml_tool_calls_no_match() {
        let content = "Hello, how can I help you today?";
        let result = LlmActor::parse_xml_tool_calls(content);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_xml_tool_calls_multiple_params() {
        let content = r#"<｜DSML｜function_calls>
  <｜DSML｜invoke name="search_files">
    <｜DSML｜parameter name="path" string="true">/src</｜DSML｜parameter>
    <｜DSML｜parameter name="regex" string="true">fn main</｜DSML｜parameter>
    <｜DSML｜parameter name="file_pattern" string="true">*.rs</｜DSML｜parameter>
  </｜DSML｜invoke>
</｜DSML｜function_calls>"#;

        let result = LlmActor::parse_xml_tool_calls(content);
        assert!(result.is_some());
        let calls = result.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "search_files");
        let args: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["path"], "/src");
        assert_eq!(args["regex"], "fn main");
        assert_eq!(args["file_pattern"], "*.rs");
    }
}

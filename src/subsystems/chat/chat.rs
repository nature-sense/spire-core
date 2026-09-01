// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! ChatActor — manages chat dialogs and messages in-memory.
//!
//! This actor stores chat dialogs and messages, providing CRUD operations
//! for the chat system. It is the single source of truth for chat state.

use async_trait::async_trait;
use chrono::Utc;
use std::collections::HashMap;

use crate::actors::{Actor, ActorError};

/// A chat message.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessageData {
    pub id: String,
    pub role: String,
    pub content: String,
    pub timestamp: String,
    /// Optional embedded widget (build-list, radio-group, checkbox-list, progress-bar).
    /// Stored as opaque JSON so the frontend drives rendering.
    pub widget: Option<serde_json::Value>,
}

/// A chat dialog.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatDialog {
    pub id: String,
    pub title: String,
    pub messages: Vec<ChatMessageData>,
    pub created_at: String,
    pub updated_at: String,
}

/// Messages for the Chat actor.
pub enum ChatMessage {
    /// Get the active chat dialog.
    GetActive {
        reply_to: tokio::sync::oneshot::Sender<Option<ChatDialog>>,
    },
    /// Get all chat dialogs.
    GetHistory {
        reply_to: tokio::sync::oneshot::Sender<Vec<ChatDialog>>,
    },
    /// Append a message to a chat dialog.
    Append {
        chat_id: String,
        content: String,
        role: String,
        widget: Option<serde_json::Value>,
        reply_to: tokio::sync::oneshot::Sender<Result<ChatMessageData, ActorError>>,
    },
    /// Clear all messages in a chat dialog.
    Clear {
        chat_id: String,
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
    /// Set the title of a chat dialog.
    SetTitle {
        chat_id: String,
        title: String,
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
    /// Update the state of a widget embedded in a message.
    UpdateWidget {
        widget_id: String,
        state: serde_json::Value,
        reply_to: tokio::sync::oneshot::Sender<Result<(), ActorError>>,
    },
}

/// Actor that manages chat dialogs and messages.
pub struct ChatActor {
    dialogs: HashMap<String, ChatDialog>,
    active_id: Option<String>,
}

impl ChatActor {
    pub fn new() -> Self {
        let mut dialogs = HashMap::new();
        let now = Utc::now().to_rfc3339();
        let default_id = "default".to_string();
        dialogs.insert(
            default_id.clone(),
            ChatDialog {
                id: default_id.clone(),
                title: "New Chat".to_string(),
                messages: vec![],
                created_at: now.clone(),
                updated_at: now,
            },
        );
        Self {
            dialogs,
            active_id: Some(default_id),
        }
    }
}

impl Default for ChatActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for ChatActor {
    type Message = ChatMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            ChatMessage::GetActive { reply_to } => {
                let dialog = self
                    .active_id
                    .as_ref()
                    .and_then(|id| self.dialogs.get(id).cloned());
                let _ = reply_to.send(dialog);
            }
            ChatMessage::GetHistory { reply_to } => {
                let dialogs: Vec<ChatDialog> = self.dialogs.values().cloned().collect();
                let _ = reply_to.send(dialogs);
            }
            ChatMessage::Append {
                chat_id,
                content,
                role,
                widget,
                reply_to,
            } => {
                let result = self.append_message(&chat_id, &content, &role, widget);
                let _ = reply_to.send(result);
            }
            ChatMessage::Clear { chat_id, reply_to } => {
                if let Some(dialog) = self.dialogs.get_mut(&chat_id) {
                    dialog.messages.clear();
                    dialog.updated_at = Utc::now().to_rfc3339();
                    let _ = reply_to.send(Ok(()));
                } else {
                    let _ = reply_to.send(Err(ActorError::Internal(format!(
                        "Chat not found: {}",
                        chat_id
                    ))));
                }
            }
            ChatMessage::SetTitle {
                chat_id,
                title,
                reply_to,
            } => {
                if let Some(dialog) = self.dialogs.get_mut(&chat_id) {
                    dialog.title = title;
                    dialog.updated_at = Utc::now().to_rfc3339();
                    let _ = reply_to.send(Ok(()));
                } else {
                    let _ = reply_to.send(Err(ActorError::Internal(format!(
                        "Chat not found: {}",
                        chat_id
                    ))));
                }
            }
            ChatMessage::UpdateWidget {
                widget_id,
                state,
                reply_to,
            } => {
                let result = self.update_widget(&widget_id, state);
                let _ = reply_to.send(result);
            }
        }
    }
}

impl ChatActor {
    fn append_message(
        &mut self,
        chat_id: &str,
        content: &str,
        role: &str,
        widget: Option<serde_json::Value>,
    ) -> Result<ChatMessageData, ActorError> {
        let dialog = self
            .dialogs
            .get_mut(chat_id)
            .ok_or_else(|| ActorError::Internal(format!("Chat not found: {}", chat_id)))?;

        let message = ChatMessageData {
            id: format!("msg-{}", uuid::Uuid::new_v4()),
            role: role.to_string(),
            content: content.to_string(),
            timestamp: Utc::now().to_rfc3339(),
            widget,
        };

        dialog.messages.push(message.clone());
        dialog.updated_at = Utc::now().to_rfc3339();

        Ok(message)
    }

    /// Update the state of a widget by finding the message that contains it.
    /// The widget_id is matched against the "widgetId" field inside the widget JSON.
    fn update_widget(
        &mut self,
        widget_id: &str,
        new_state: serde_json::Value,
    ) -> Result<(), ActorError> {
        for dialog in self.dialogs.values_mut() {
            for msg in &mut dialog.messages {
                if let Some(ref mut widget) = msg.widget {
                    if let Some(current_id) = widget.get("widgetId").and_then(|v| v.as_str()) {
                        if current_id == widget_id {
                            // Update the "state" field inside the widget JSON
                            if let Some(obj) = widget.as_object_mut() {
                                obj.insert("state".to_string(), new_state.clone());
                            }
                            dialog.updated_at = Utc::now().to_rfc3339();
                            return Ok(());
                        }
                    }
                }
            }
        }
        Err(ActorError::Internal(format!(
            "Widget not found: {}",
            widget_id
        )))
    }
}

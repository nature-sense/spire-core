// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! TransportActor — JSON-RPC 2.0 transport over TCP socket as an actor.
//!
//! This module provides bidirectional JSON-RPC communication over a TCP
//! loopback socket, replacing the previous `Arc<Mutex<Transport>>` design
//! with a proper actor that owns all state directly.
//!
//! Architecture:
//!   - Core binds to 127.0.0.1:0 (OS-assigned port)
//!   - Core prints "SPIRE_PORT=<port>" to stdout for the extension to read
//!   - Extension connects to 127.0.0.1:<port>
//!   - All JSON-RPC messages flow over the TCP connection
//!
//! Protocol: https://www.jsonrpc.org/specification
//!
//! Messages (newline-delimited JSON):
//!   {"jsonrpc":"2.0","id":1,"method":"chat/getActive","params":{}}
//!   {"jsonrpc":"2.0","id":1,"result":{...}}
//!   {"jsonrpc":"2.0","method":"event/chat/message","params":{...}}

use async_trait::async_trait;
use std::collections::HashMap;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

use crate::actors::Actor;

/// A notification received from the extension, forwarded to the coordinator.
pub struct IncomingNotification {
    pub method: String,
    pub params: serde_json::Value,
}

/// A pending outgoing request (core → extension) waiting for a response.
struct PendingRequest {
    response_tx: oneshot::Sender<Result<serde_json::Value, String>>,
}

/// Messages for the TransportActor.
pub enum TransportMessage {
    /// Bind to a loopback port and start listening.
    Bind {
        reply_to: oneshot::Sender<Result<u16, String>>,
    },
    /// Accept the extension's TCP connection.
    Accept {
        reply_to: oneshot::Sender<Result<(), String>>,
    },
    /// Send a JSON-RPC request to the extension and wait for a response.
    CallExtension {
        method: String,
        params: serde_json::Value,
        reply_to: oneshot::Sender<Result<serde_json::Value, String>>,
    },
    /// Send a JSON-RPC response to the extension (for incoming requests).
    SendResponse { id: u64, result: serde_json::Value },
    /// Send a JSON-RPC error response to the extension.
    SendError { id: u64, code: i64, message: String },
    /// Send a notification (event) to the extension.
    SendNotification {
        method: String,
        params: serde_json::Value,
    },
    /// A raw line was received from the socket (from the reader task).
    SocketLine { line: String },
    /// Register a handler for incoming requests from the extension.
    /// The handler is a sender to the coordinator actor.
    SetRequestHandler {
        handler_tx: mpsc::Sender<IncomingRequestMessage>,
    },
    /// Register a handler for incoming notifications from the extension.
    /// The handler is a sender to the coordinator actor.
    SetNotificationHandler {
        notification_tx: mpsc::Sender<IncomingNotification>,
    },
    /// Set the transport actor's own sender (so the reader task can send messages back).
    SetSelfTx {
        self_tx: mpsc::Sender<TransportMessage>,
    },
    /// Remove a pending request entry (for timeout cleanup by spawned tasks).
    /// Prevents memory leaks when call_extension timed out and the response
    /// has not yet arrived — the spawned task sends this to clean up.
    RemovePending { id: u64 },
}

/// A message representing an incoming JSON-RPC request from the extension.
pub struct IncomingRequestMessage {
    pub id: u64,
    pub method: String,
    pub params: serde_json::Value,
    /// Sender to send the response back through the transport.
    pub response_tx: oneshot::Sender<serde_json::Value>,
}

/// Actor that owns the TCP socket and manages JSON-RPC communication.
pub struct TransportActor {
    /// Pending outgoing requests awaiting responses.
    pending: HashMap<u64, PendingRequest>,
    /// Next request ID for outgoing requests.
    next_id: u64,
    /// The TCP listener (held to keep the port bound).
    listener: Option<TcpListener>,
    /// Write half of the accepted connection.
    writer: Option<tokio::io::WriteHalf<TcpStream>>,
    /// Join handle for the socket reader task.
    _reader_handle: Option<tokio::task::JoinHandle<()>>,
    /// Sender for incoming requests — forwarded to the coordinator.
    handler_tx: Option<mpsc::Sender<IncomingRequestMessage>>,
    /// Sender for incoming notifications — forwarded to the coordinator.
    notification_tx: Option<mpsc::Sender<IncomingNotification>>,
    /// Our own sender (set via SetSelfTx) — used by the reader task to send lines back.
    self_tx: Option<mpsc::Sender<TransportMessage>>,
}

impl Default for TransportActor {
    fn default() -> Self {
        Self::new()
    }
}

impl TransportActor {
    pub fn new() -> Self {
        Self {
            pending: HashMap::new(),
            next_id: 1,
            listener: None,
            writer: None,
            _reader_handle: None,
            handler_tx: None,
            notification_tx: None,
            self_tx: None,
        }
    }

    /// Write a JSON message to the socket.
    async fn write_json(&mut self, value: &serde_json::Value) {
        let json = serde_json::to_string(value).unwrap_or_default();
        if let Some(ref mut w) = self.writer {
            let line = format!("{}\n", json);
            if let Err(e) = w.write_all(line.as_bytes()).await {
                warn!("TransportActor: failed to write to socket: {}", e);
            }
        } else {
            debug!("TransportActor: cannot write JSON, socket writer not available (not yet connected)");
        }
    }

    /// Process a single line received from the socket.
    async fn process_line(&mut self, line: &str) {
        let msg: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                error!("TransportActor: failed to parse JSON: {}", e);
                return;
            }
        };

        let id = msg.get("id").and_then(|v| v.as_u64());
        let method = msg
            .get("method")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        match (id, method) {
            // Notification (no id) — forward to the notification handler
            (None, Some(method_name)) => {
                debug!("TransportActor: received notification: {}", method_name);
                if let Some(ref notification_tx) = self.notification_tx {
                    let params = msg
                        .get("params")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    let _ = notification_tx.try_send(IncomingNotification {
                        method: method_name,
                        params,
                    });
                }
            }
            // Response to one of our outgoing requests
            (Some(id_val), None) => {
                if let Some(pending_req) = self.pending.remove(&id_val) {
                    if let Some(error) = msg.get("error") {
                        let msg_str = error
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown error")
                            .to_string();
                        let _ = pending_req.response_tx.send(Err(msg_str));
                    } else {
                        let result = msg
                            .get("result")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        let _ = pending_req.response_tx.send(Ok(result));
                    }
                } else {
                    warn!(
                        "TransportActor: received response for unknown request id={}",
                        id_val
                    );
                }
            }
            // Incoming request from the extension
            (Some(id_val), Some(method_name)) => {
                if let Some(ref handler_tx) = self.handler_tx {
                    let params = msg
                        .get("params")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    let (response_tx, response_rx) = oneshot::channel();
                    let handler_tx = handler_tx.clone();
                    let self_tx = self.self_tx.clone();

                    // Spawn a task to forward to the handler and send the response back
                    tokio::spawn(async move {
                        if handler_tx
                            .send(IncomingRequestMessage {
                                id: id_val,
                                method: method_name.to_string(),
                                params,
                                response_tx,
                            })
                            .await
                            .is_err()
                        {
                            error!("TransportActor: failed to forward incoming request to handler");
                            return;
                        }

                        // Wait for the handler's response
                        match response_rx.await {
                            Ok(result) => {
                                // Send the response back to the transport actor for writing
                                if let Some(ref tx) = self_tx {
                                    let _ = tx
                                        .send(TransportMessage::SendResponse { id: id_val, result })
                                        .await;
                                }
                            }
                            Err(_) => {
                                error!(
                                    "TransportActor: handler response channel closed for id={}",
                                    id_val
                                );
                            }
                        }
                    });
                } else {
                    warn!(
                        "TransportActor: no request handler registered, sending error for id={}",
                        id_val
                    );
                    let response = serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id_val,
                        "error": {
                            "code": -32601,
                            "message": "Method not found: no handler registered",
                        },
                    });
                    self.write_json(&response).await;
                }
            }
            (None, None) => {
                warn!("TransportActor: received invalid message (no id and no method)");
            }
        }
    }
}

#[async_trait]
impl Actor for TransportActor {
    type Message = TransportMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            TransportMessage::Bind { reply_to } => match TcpListener::bind("127.0.0.1:0").await {
                Ok(listener) => {
                    let local_addr = match listener.local_addr() {
                        Ok(addr) => addr,
                        Err(e) => {
                            let _ = reply_to.send(Err(format!("Failed to get local addr: {}", e)));
                            return;
                        }
                    };
                    let port = local_addr.port();
                    info!("TransportActor: listening on 127.0.0.1:{}", port);
                    self.listener = Some(listener);
                    let _ = reply_to.send(Ok(port));
                }
                Err(e) => {
                    let _ = reply_to.send(Err(format!("Failed to bind: {}", e)));
                }
            },

            TransportMessage::Accept { reply_to } => {
                let listener = match self.listener.as_ref() {
                    Some(l) => l,
                    None => {
                        let _ = reply_to.send(Err("Transport not bound yet".to_string()));
                        return;
                    }
                };

                // We need to accept without holding a reference to self.listener
                // that would prevent us from moving it. We take ownership.
                // But we can't take ownership from an Option reference.
                // Solution: use try_clone or just accept on the original listener.
                // Actually, TcpListener::accept takes &self, so we can use it via reference.
                match listener.accept().await {
                    Ok((stream, peer_addr)) => {
                        info!("TransportActor: accepted connection from {}", peer_addr);

                        let (reader, writer_half) = tokio::io::split(stream);
                        self.writer = Some(writer_half);

                        // Spawn a task to read lines from the socket.
                        // The reader sends lines back to the actor via its mailbox.
                        let self_tx = match self.self_tx.clone() {
                            Some(tx) => tx,
                            None => {
                                error!("TransportActor: self_tx not set before Accept");
                                let _ = reply_to.send(Err("self_tx not set".to_string()));
                                return;
                            }
                        };

                        let _reader_handle = tokio::spawn(async move {
                            let mut buf_reader = BufReader::new(reader);
                            let mut line = String::new();
                            loop {
                                line.clear();
                                match buf_reader.read_line(&mut line).await {
                                    Ok(0) => {
                                        info!("TransportActor: socket EOF (extension closed connection)");
                                        break;
                                    }
                                    Ok(_) => {
                                        let trimmed = line.trim().to_string();
                                        if !trimmed.is_empty()
                                            && self_tx
                                                .send(TransportMessage::SocketLine {
                                                    line: trimmed,
                                                })
                                                .await
                                                .is_err()
                                        {
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        error!("TransportActor: error reading from socket: {}", e);
                                        break;
                                    }
                                }
                            }
                        });

                        self._reader_handle = Some(_reader_handle);
                        let _ = reply_to.send(Ok(()));
                    }
                    Err(e) => {
                        let _ = reply_to.send(Err(format!("Failed to accept: {}", e)));
                    }
                }
            }

            TransportMessage::SocketLine { line } => {
                self.process_line(&line).await;
            }

            TransportMessage::CallExtension {
                method,
                params,
                reply_to,
            } => {
                let id = self.next_id;
                self.next_id += 1;

                let (response_tx, response_rx) = oneshot::channel();

                self.pending.insert(id, PendingRequest { response_tx });

                let request = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "method": method,
                    "params": params,
                });
                self.write_json(&request).await;

                // Spawn a task to await the response, freeing the actor's mailbox
                // to process incoming messages (including the response itself).
                // This prevents the re-entrancy deadlock where the actor mailbox
                // blocks waiting for a response that the reader task can't deliver.
                let self_tx = self.self_tx.clone();
                tokio::spawn(async move {
                    match tokio::time::timeout(std::time::Duration::from_secs(30), response_rx)
                        .await
                    {
                        Ok(Ok(result)) => {
                            // result is Result<Value, String> from PendingRequest.response_tx
                            let _ = reply_to.send(result);
                        }
                        Ok(Err(_)) => {
                            // Signal cleanup: remove pending entry and notify caller
                            if let Some(ref tx) = self_tx {
                                let _ = tx.send(TransportMessage::RemovePending { id }).await;
                            }
                            let _ = reply_to
                                .send(Err(format!("Response channel closed for '{}'", method)));
                        }
                        Err(_) => {
                            // Timeout: signal cleanup and notify caller
                            if let Some(ref tx) = self_tx {
                                let _ = tx.send(TransportMessage::RemovePending { id }).await;
                            }
                            let _ = reply_to
                                .send(Err(format!("Request timed out: {} (id={})", method, id)));
                        }
                    }
                });
            }

            TransportMessage::RemovePending { id } => {
                self.pending.remove(&id);
                debug!("TransportActor: cleaned up pending request id={}", id);
            }

            TransportMessage::SendResponse { id, result } => {
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": result,
                });
                self.write_json(&response).await;
            }

            TransportMessage::SendError { id, code, message } => {
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": code,
                        "message": message,
                    },
                });
                self.write_json(&response).await;
            }

            TransportMessage::SendNotification { method, params } => {
                let notification = serde_json::json!({
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params,
                });
                self.write_json(&notification).await;
            }

            TransportMessage::SetRequestHandler { handler_tx } => {
                self.handler_tx = Some(handler_tx);
                info!("TransportActor: request handler registered");
            }

            TransportMessage::SetNotificationHandler { notification_tx } => {
                self.notification_tx = Some(notification_tx);
                info!("TransportActor: notification handler registered");
            }

            TransportMessage::SetSelfTx { self_tx } => {
                self.self_tx = Some(self_tx);
            }
        }
    }
}

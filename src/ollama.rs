use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::audit::{self, AuditHandle, AuditRecord, SCHEMA_VERSION};
use crate::chat::{ChatMessage, MessageCorrelation};
use crate::incoming::MessageSource;

const OLLAMA_URL: &str = "http://127.0.0.1:11434";

fn correlated_chat_message(
    content: String,
    from: Option<String>,
    conversation_id: &str,
    request_id: &str,
) -> ChatMessage {
    ChatMessage {
        content,
        from,
        correlation: Some(MessageCorrelation {
            conversation_id: conversation_id.to_string(),
            event_id: audit::new_id(),
            request_id: request_id.to_string(),
            timestamp_rfc3339: audit::now_rfc3339(),
        }),
        source: MessageSource::System,
        api_auto_respond: false,
    }
}

#[derive(Clone)]
pub struct OllamaController {
    status: Arc<Mutex<OllamaStatus>>,
    models: Arc<Mutex<Vec<String>>>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum OllamaStatus {
    Running,
    Stopped,
    Checking,
}

impl OllamaController {
    pub fn new() -> Self {
        Self {
            status: Arc::new(Mutex::new(OllamaStatus::Stopped)),
            models: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn status(&self) -> OllamaStatus {
        *self.status.lock().unwrap()
    }

    pub fn models(&self) -> Vec<String> {
        self.models.lock().unwrap().clone()
    }

    /// Trigger an async status check for Ollama and refresh models if running
    pub fn check_status(&self) {
        let status = self.status.clone();
        let models = self.models.clone();
        std::thread::spawn(move || {
            *status.lock().unwrap() = OllamaStatus::Checking;

            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(2))
                .build();

            let is_running = if let Ok(client) = client {
                client
                    .get(format!("{}/api/tags", OLLAMA_URL))
                    .send()
                    .is_ok()
            } else {
                false
            };

            let new_status = if is_running {
                OllamaStatus::Running
            } else {
                OllamaStatus::Stopped
            };
            *status.lock().unwrap() = new_status;

            if new_status == OllamaStatus::Running {
                fetch_models_inner(models);
            }
        });
    }

    /// Fetch available Ollama models
    pub fn fetch_models(&self) {
        let models = self.models.clone();
        std::thread::spawn(move || {
            fetch_models_inner(models);
        });
    }

    /// Send a message to Ollama using the selected model.
    /// `send_fn` is used to push messages into the UI inbox.
    pub fn send_message(
        &self,
        model: String,
        message: String,
        num_predict: Option<i32>,
        audit_handle: Arc<AuditHandle>,
        conversation_id: String,
        request_id: String,
        send_fn: Box<dyn Fn(ChatMessage) + Send + Sync>,
    ) {
        let model_clone = model.clone();
        std::thread::spawn(move || {
            let start_event_id = audit::new_id();
            let options_json = match num_predict {
                Some(limit) => serde_json::json!({ "num_predict": limit }),
                None => serde_json::json!({}),
            };

            tracing::info!(
                request_id = %request_id,
                model = %model_clone,
                num_predict = ?num_predict,
                message_len = message.len(),
                "ollama request start"
            );

            let start_record = AuditRecord {
                schema_version: SCHEMA_VERSION,
                kind: "ollama_start",
                ts: audit::now_rfc3339(),
                conversation_id: conversation_id.clone(),
                request_id: request_id.clone(),
                event_id: start_event_id.clone(),
                details: serde_json::json!({
                    "model": model_clone,
                    "endpoint": format!("{}/api/chat", OLLAMA_URL),
                    "num_predict": num_predict,
                    "options": options_json,
                    "prompt_len": message.len(),
                }),
            };
            if let Err(e) = audit_handle.append_json_line(&start_record) {
                tracing::warn!(error = %e, request_id = %request_id, "audit append failed");
            }

            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(60))
                .build();

            if let Ok(client) = client {
                let mut request_body = serde_json::json!({
                    "model": model_clone,
                    "messages": [
                        {
                            "role": "user",
                            "content": message
                        }
                    ],
                    "stream": false
                });

                if let Some(limit) = num_predict {
                    request_body["options"] = serde_json::json!({
                        "num_predict": limit
                    });
                }

                tracing::debug!(
                    request_id = %request_id,
                    url = %format!("{}/api/chat", OLLAMA_URL),
                    "ollama HTTP POST"
                );

                let response_result = client
                    .post(format!("{}/api/chat", OLLAMA_URL))
                    .json(&request_body)
                    .send();

                match response_result {
                    Ok(response) => {
                        let status = response.status();
                        let status_u16 = status.as_u16();
                        tracing::info!(
                            request_id = %request_id,
                            status = status_u16,
                            "ollama HTTP response status"
                        );

                        if !status.is_success() {
                            let error_text = response.text().ok();
                            tracing::warn!(
                                request_id = %request_id,
                                status = status_u16,
                                body = ?error_text.as_ref().map(|s| s.len()),
                                "ollama HTTP error status"
                            );
                            let end_record = AuditRecord {
                                schema_version: SCHEMA_VERSION,
                                kind: "ollama_end",
                                ts: audit::now_rfc3339(),
                                conversation_id: conversation_id.clone(),
                                request_id: request_id.clone(),
                                event_id: audit::new_id(),
                                details: serde_json::json!({
                                    "http_status": status_u16,
                                    "success": false,
                                    "error_body_len": error_text.as_ref().map(|s| s.len()),
                                }),
                            };
                            let _ = audit_handle.append_json_line(&end_record);

                            let error_msg = correlated_chat_message(
                                format!("Error: HTTP request failed with status {}", status),
                                Some("System".to_string()),
                                &conversation_id,
                                &request_id,
                            );
                            send_fn(error_msg);
                            return;
                        }

                        let response_text = match response.text() {
                            Ok(text) => text,
                            Err(e) => {
                                tracing::warn!(request_id = %request_id, error = %e, "ollama read body failed");
                                let end_record = AuditRecord {
                                    schema_version: SCHEMA_VERSION,
                                    kind: "ollama_end",
                                    ts: audit::now_rfc3339(),
                                    conversation_id: conversation_id.clone(),
                                    request_id: request_id.clone(),
                                    event_id: audit::new_id(),
                                    details: serde_json::json!({
                                        "http_status": status_u16,
                                        "success": false,
                                        "read_error": e.to_string(),
                                    }),
                                };
                                let _ = audit_handle.append_json_line(&end_record);

                                let error_msg = correlated_chat_message(
                                    format!("Error: Failed to read response: {}", e),
                                    Some("System".to_string()),
                                    &conversation_id,
                                    &request_id,
                                );
                                send_fn(error_msg);
                                return;
                            }
                        };

                        tracing::debug!(
                            request_id = %request_id,
                            len = response_text.len(),
                            "ollama response body"
                        );

                        match serde_json::from_str::<serde_json::Value>(&response_text) {
                            Ok(json) => {
                                if let Some(error) = json.get("error") {
                                    let error_msg = if let Some(error_str) = error.as_str() {
                                        format!("Error: {}", error_str)
                                    } else {
                                        format!("Error: {}", error)
                                    };
                                    tracing::warn!(
                                        request_id = %request_id,
                                        error = %error_msg,
                                        "ollama error field in JSON"
                                    );
                                    let end_record = AuditRecord {
                                        schema_version: SCHEMA_VERSION,
                                        kind: "ollama_end",
                                        ts: audit::now_rfc3339(),
                                        conversation_id: conversation_id.clone(),
                                        request_id: request_id.clone(),
                                        event_id: audit::new_id(),
                                        details: serde_json::json!({
                                            "http_status": status_u16,
                                            "success": false,
                                            "ollama_error": error_msg.clone(),
                                        }),
                                    };
                                    let _ = audit_handle.append_json_line(&end_record);

                                    let chat_error = correlated_chat_message(
                                        error_msg,
                                        Some("System".to_string()),
                                        &conversation_id,
                                        &request_id,
                                    );
                                    send_fn(chat_error);
                                    return;
                                }

                                if let Some(message_obj) = json.get("message") {
                                    if let Some(content_value) = message_obj.get("content") {
                                        if let Some(content) = content_value.as_str() {
                                            if content.is_empty() {
                                                let alternative_content = message_obj
                                                    .get("thinking")
                                                    .and_then(|t| t.as_str())
                                                    .or_else(|| json.get("response").and_then(|r| r.as_str()));

                                                if let Some(response_text) = alternative_content {
                                                    let field_name = if message_obj.get("thinking").is_some() {
                                                        "thinking"
                                                    } else {
                                                        "response"
                                                    };
                                                    tracing::debug!(
                                                        request_id = %request_id,
                                                        field = field_name,
                                                        len = response_text.len(),
                                                        "ollama using alternate content field"
                                                    );
                                                    let from_text = format!("Ollama {}", model_clone);
                                                    let ollama_msg = correlated_chat_message(
                                                        response_text.to_string(),
                                                        Some(from_text),
                                                        &conversation_id,
                                                        &request_id,
                                                    );
                                                    let end_record = AuditRecord {
                                                        schema_version: SCHEMA_VERSION,
                                                        kind: "ollama_end",
                                                        ts: audit::now_rfc3339(),
                                                        conversation_id: conversation_id.clone(),
                                                        request_id: request_id.clone(),
                                                        event_id: audit::new_id(),
                                                        details: serde_json::json!({
                                                            "http_status": status_u16,
                                                            "success": true,
                                                            "assistant_content_len": response_text.len(),
                                                            "content_field": field_name,
                                                        }),
                                                    };
                                                    let _ = audit_handle.append_json_line(&end_record);
                                                    send_fn(ollama_msg);
                                                } else {
                                                    tracing::warn!(
                                                        request_id = %request_id,
                                                        "ollama empty content"
                                                    );
                                                    let end_record = AuditRecord {
                                                        schema_version: SCHEMA_VERSION,
                                                        kind: "ollama_end",
                                                        ts: audit::now_rfc3339(),
                                                        conversation_id: conversation_id.clone(),
                                                        request_id: request_id.clone(),
                                                        event_id: audit::new_id(),
                                                        details: serde_json::json!({
                                                            "http_status": status_u16,
                                                            "success": false,
                                                            "reason": "empty_content",
                                                        }),
                                                    };
                                                    let _ = audit_handle.append_json_line(&end_record);

                                                    let error_msg = correlated_chat_message(
                                                        "Error: Empty response from Ollama".to_string(),
                                                        Some("System".to_string()),
                                                        &conversation_id,
                                                        &request_id,
                                                    );
                                                    send_fn(error_msg);
                                                }
                                            } else {
                                                tracing::info!(
                                                    request_id = %request_id,
                                                    len = content.len(),
                                                    "ollama assistant content"
                                                );
                                                let from_text = format!("Ollama {}", model_clone);
                                                let ollama_msg = correlated_chat_message(
                                                    content.to_string(),
                                                    Some(from_text),
                                                    &conversation_id,
                                                    &request_id,
                                                );
                                                let end_record = AuditRecord {
                                                    schema_version: SCHEMA_VERSION,
                                                    kind: "ollama_end",
                                                    ts: audit::now_rfc3339(),
                                                    conversation_id: conversation_id.clone(),
                                                    request_id: request_id.clone(),
                                                    event_id: audit::new_id(),
                                                    details: serde_json::json!({
                                                        "http_status": status_u16,
                                                        "success": true,
                                                        "assistant_content_len": content.len(),
                                                        "content_field": "content",
                                                    }),
                                                };
                                                let _ = audit_handle.append_json_line(&end_record);
                                                send_fn(ollama_msg);
                                            }
                                        } else {
                                            tracing::warn!(
                                                request_id = %request_id,
                                                "ollama content not a string"
                                            );
                                            let end_record = AuditRecord {
                                                schema_version: SCHEMA_VERSION,
                                                kind: "ollama_end",
                                                ts: audit::now_rfc3339(),
                                                conversation_id: conversation_id.clone(),
                                                request_id: request_id.clone(),
                                                event_id: audit::new_id(),
                                                details: serde_json::json!({
                                                    "http_status": status_u16,
                                                    "success": false,
                                                    "reason": "invalid_content_type",
                                                }),
                                            };
                                            let _ = audit_handle.append_json_line(&end_record);

                                            let error_msg = correlated_chat_message(
                                                "Error: Invalid content format from Ollama".to_string(),
                                                Some("System".to_string()),
                                                &conversation_id,
                                                &request_id,
                                            );
                                            send_fn(error_msg);
                                        }
                                    } else {
                                        tracing::warn!(request_id = %request_id, "ollama missing content");
                                        let end_record = AuditRecord {
                                            schema_version: SCHEMA_VERSION,
                                            kind: "ollama_end",
                                            ts: audit::now_rfc3339(),
                                            conversation_id: conversation_id.clone(),
                                            request_id: request_id.clone(),
                                            event_id: audit::new_id(),
                                            details: serde_json::json!({
                                                "http_status": status_u16,
                                                "success": false,
                                                "reason": "missing_content",
                                            }),
                                        };
                                        let _ = audit_handle.append_json_line(&end_record);

                                        let error_msg = correlated_chat_message(
                                            "Error: Invalid response format from Ollama".to_string(),
                                            Some("System".to_string()),
                                            &conversation_id,
                                            &request_id,
                                        );
                                        send_fn(error_msg);
                                    }
                                } else {
                                    tracing::warn!(request_id = %request_id, "ollama missing message field");
                                    let end_record = AuditRecord {
                                        schema_version: SCHEMA_VERSION,
                                        kind: "ollama_end",
                                        ts: audit::now_rfc3339(),
                                        conversation_id: conversation_id.clone(),
                                        request_id: request_id.clone(),
                                        event_id: audit::new_id(),
                                        details: serde_json::json!({
                                            "http_status": status_u16,
                                            "success": false,
                                            "reason": "missing_message",
                                        }),
                                    };
                                    let _ = audit_handle.append_json_line(&end_record);

                                    let error_msg = correlated_chat_message(
                                        "Error: Invalid response format from Ollama".to_string(),
                                        Some("System".to_string()),
                                        &conversation_id,
                                        &request_id,
                                    );
                                    send_fn(error_msg);
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    request_id = %request_id,
                                    error = %e,
                                    "ollama JSON parse failed"
                                );
                                let end_record = AuditRecord {
                                    schema_version: SCHEMA_VERSION,
                                    kind: "ollama_end",
                                    ts: audit::now_rfc3339(),
                                    conversation_id: conversation_id.clone(),
                                    request_id: request_id.clone(),
                                    event_id: audit::new_id(),
                                    details: serde_json::json!({
                                        "http_status": status_u16,
                                        "success": false,
                                        "parse_error": e.to_string(),
                                        "response_preview_len": response_text.len(),
                                    }),
                                };
                                let _ = audit_handle.append_json_line(&end_record);

                                let error_msg = correlated_chat_message(
                                    format!("Error: Failed to parse Ollama response: {}", e),
                                    Some("System".to_string()),
                                    &conversation_id,
                                    &request_id,
                                );
                                send_fn(error_msg);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(request_id = %request_id, error = %e, "ollama HTTP request failed");
                        let end_record = AuditRecord {
                            schema_version: SCHEMA_VERSION,
                            kind: "ollama_end",
                            ts: audit::now_rfc3339(),
                            conversation_id: conversation_id.clone(),
                            request_id: request_id.clone(),
                            event_id: audit::new_id(),
                            details: serde_json::json!({
                                "http_status": null,
                                "success": false,
                                "transport_error": e.to_string(),
                            }),
                        };
                        let _ = audit_handle.append_json_line(&end_record);

                        let error_msg = correlated_chat_message(
                            format!("Error: HTTP request failed: {}", e),
                            Some("System".to_string()),
                            &conversation_id,
                            &request_id,
                        );
                        send_fn(error_msg);
                    }
                }
            } else {
                tracing::error!(request_id = %request_id, "ollama failed to create HTTP client");
                let end_record = AuditRecord {
                    schema_version: SCHEMA_VERSION,
                    kind: "ollama_end",
                    ts: audit::now_rfc3339(),
                    conversation_id: conversation_id.clone(),
                    request_id: request_id.clone(),
                    event_id: audit::new_id(),
                    details: serde_json::json!({
                        "success": false,
                        "reason": "client_build_failed",
                    }),
                };
                let _ = audit_handle.append_json_line(&end_record);
            }
        });
    }
}

impl Default for OllamaController {
    fn default() -> Self {
        Self::new()
    }
}

fn fetch_models_inner(models: Arc<Mutex<Vec<String>>>) {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build();

    let model_list = if let Ok(client) = client {
        if let Ok(response) = client.get(format!("{}/api/tags", OLLAMA_URL)).send() {
            if let Ok(json) = response.json::<serde_json::Value>() {
                if let Some(models_array) = json.get("models").and_then(|m| m.as_array()) {
                    models_array
                        .iter()
                        .filter_map(|m| {
                            m.get("name")
                                .and_then(|n| n.as_str())
                                .map(|s| s.to_string())
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    *models.lock().unwrap() = model_list;
}

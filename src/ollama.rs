use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::audit::{self, AuditHandle, AuditRecord, SCHEMA_VERSION};
use crate::chat::{AssistantGeneration, ChatMessage, MessageCorrelation};
use crate::incoming::MessageSource;

pub const OLLAMA_URL: &str = "http://127.0.0.1:11434";

/// One turn in an Ollama `/api/chat` `messages` array (`system` | `user` | `assistant`).
#[derive(Clone, Debug)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

/// Generation options passed to Ollama under `options` (local inference only).
#[derive(Clone, Debug, Default)]
pub struct OllamaChatOptions {
    pub num_predict: Option<i32>,
    pub temperature: Option<f64>,
    pub seed: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct ParsedAssistant {
    pub content: String,
    pub prompt_eval_count: Option<u64>,
    pub eval_count: Option<u64>,
}

fn build_options_json(opts: &OllamaChatOptions) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    if let Some(n) = opts.num_predict {
        map.insert("num_predict".into(), n.into());
    }
    if let Some(t) = opts.temperature {
        map.insert("temperature".into(), serde_json::json!(t));
    }
    if let Some(s) = opts.seed {
        map.insert("seed".into(), s.into());
    }
    serde_json::Value::Object(map)
}

fn messages_to_json(messages: &[OllamaMessage]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|m| {
            serde_json::json!({
                "role": m.role,
                "content": m.content
            })
        })
        .collect()
}

/// Blocking call to local Ollama `/api/chat` (no audit).
pub fn chat_completion_sync(
    model: &str,
    messages: &[OllamaMessage],
    options: &OllamaChatOptions,
) -> Result<ParsedAssistant, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|_| "failed to build HTTP client".to_string())?;

    let mut request_body = serde_json::json!({
        "model": model,
        "messages": messages_to_json(messages),
        "stream": false
    });
    let opts = build_options_json(options);
    if !opts.as_object().map_or(true, serde_json::Map::is_empty) {
        request_body["options"] = opts;
    }

    let response = client
        .post(format!("{}/api/chat", OLLAMA_URL))
        .json(&request_body)
        .send()
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    let status = response.status();
    let status_u16 = status.as_u16();
    if !status.is_success() {
        let _ = response.text();
        return Err(format!("Ollama returned HTTP {}", status_u16));
    }

    let response_text = response
        .text()
        .map_err(|e| format!("failed to read response body: {}", e))?;

    let json: serde_json::Value = serde_json::from_str(&response_text)
        .map_err(|e| format!("failed to parse Ollama JSON: {}", e))?;

    parse_ollama_chat_value(&json, model)
}

fn parse_ollama_chat_value(json: &serde_json::Value, _model: &str) -> Result<ParsedAssistant, String> {
    let prompt_eval_count = json.get("prompt_eval_count").and_then(|v| v.as_u64());
    let eval_count = json.get("eval_count").and_then(|v| v.as_u64());

    if let Some(error) = json.get("error") {
        let error_msg = if let Some(error_str) = error.as_str() {
            format!("{}", error_str)
        } else {
            format!("{}", error)
        };
        return Err(error_msg);
    }

    let message_obj = json
        .get("message")
        .ok_or_else(|| "missing message field in Ollama response".to_string())?;

    let content_value = message_obj
        .get("content")
        .ok_or_else(|| "missing content in Ollama message".to_string())?;

    if let Some(content) = content_value.as_str() {
        if content.is_empty() {
            let alternative_content = message_obj
                .get("thinking")
                .and_then(|t| t.as_str())
                .or_else(|| json.get("response").and_then(|r| r.as_str()));

            if let Some(response_text) = alternative_content {
                return Ok(ParsedAssistant {
                    content: response_text.to_string(),
                    prompt_eval_count,
                    eval_count,
                });
            }
            return Err("empty response from Ollama".to_string());
        }
        return Ok(ParsedAssistant {
            content: content.to_string(),
            prompt_eval_count,
            eval_count,
        });
    }

    Err("invalid content format from Ollama".to_string())
}

/// Same as [`chat_completion_sync`] but writes `ollama_start` / `ollama_end` audit lines (shared with UI path).
pub fn chat_completion_with_audit(
    model: &str,
    messages: &[OllamaMessage],
    options: &OllamaChatOptions,
    audit_handle: Arc<AuditHandle>,
    conversation_id: String,
    request_id: String,
) -> Result<ParsedAssistant, String> {
    let start_event_id = audit::new_id();
    let options_json = build_options_json(options);

    tracing::info!(
        request_id = %request_id,
        model = %model,
        messages_len = messages.len(),
        num_predict = ?options.num_predict,
        temperature = ?options.temperature,
        seed = ?options.seed,
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
            "model": model,
            "endpoint": format!("{}/api/chat", OLLAMA_URL),
            "num_predict": options.num_predict,
            "temperature": options.temperature,
            "seed": options.seed,
            "options": options_json,
            "messages_count": messages.len(),
            "prompt_len": messages.iter().map(|m| m.content.len()).sum::<usize>(),
        }),
    };
    if let Err(e) = audit_handle.append_json_line(&start_record) {
        tracing::warn!(error = %e, request_id = %request_id, "audit append failed");
    }

    let result = chat_completion_sync(model, messages, options);

    match &result {
        Ok(parsed) => {
            let end_record = AuditRecord {
                schema_version: SCHEMA_VERSION,
                kind: "ollama_end",
                ts: audit::now_rfc3339(),
                conversation_id: conversation_id.clone(),
                request_id: request_id.clone(),
                event_id: audit::new_id(),
                details: serde_json::json!({
                    "success": true,
                    "assistant_content_len": parsed.content.len(),
                    "prompt_eval_count": parsed.prompt_eval_count,
                    "eval_count": parsed.eval_count,
                }),
            };
            let _ = audit_handle.append_json_line(&end_record);
        }
        Err(err) => {
            let end_record = AuditRecord {
                schema_version: SCHEMA_VERSION,
                kind: "ollama_end",
                ts: audit::now_rfc3339(),
                conversation_id: conversation_id.clone(),
                request_id: request_id.clone(),
                event_id: audit::new_id(),
                details: serde_json::json!({
                    "success": false,
                    "error": err,
                }),
            };
            let _ = audit_handle.append_json_line(&end_record);
        }
    }

    result
}

/// Lists model names from local Ollama `GET /api/tags`.
pub fn list_ollama_models_blocking() -> Result<Vec<String>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;

    let response = client
        .get(format!("{}/api/tags", OLLAMA_URL))
        .send()
        .map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        return Err(format!("Ollama returned HTTP {}", response.status().as_u16()));
    }

    let json: serde_json::Value = response.json().map_err(|e| e.to_string())?;
    let models_array = json
        .get("models")
        .and_then(|m| m.as_array())
        .ok_or_else(|| "invalid /api/tags response".to_string())?;

    Ok(models_array
        .iter()
        .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(|s| s.to_string()))
        .collect())
}

fn correlated_chat_message(
    content: String,
    from: Option<String>,
    conversation_id: &str,
    request_id: &str,
    assistant_generation: Option<AssistantGeneration>,
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
        assistant_generation,
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
            let messages = vec![OllamaMessage {
                role: "user".into(),
                content: message,
            }];
            let options = OllamaChatOptions {
                num_predict,
                ..Default::default()
            };

            match chat_completion_with_audit(
                &model_clone,
                &messages,
                &options,
                audit_handle,
                conversation_id.clone(),
                request_id.clone(),
            ) {
                Ok(parsed) => {
                    let from_text = format!("Ollama {}", model_clone);
                    let ollama_msg = correlated_chat_message(
                        parsed.content,
                        Some(from_text),
                        &conversation_id,
                        &request_id,
                        Some(AssistantGeneration {
                            model: model_clone,
                            num_predict,
                        }),
                    );
                    send_fn(ollama_msg);
                }
                Err(err_msg) => {
                    send_fn(correlated_chat_message(
                        format!("Error: {}", err_msg),
                        Some("System".to_string()),
                        &conversation_id,
                        &request_id,
                        None,
                    ));
                }
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

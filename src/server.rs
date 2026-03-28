use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use std::sync::mpsc;

use crate::audit::{self, AuditRecord, SCHEMA_VERSION};
use crate::chat::{ChatMessage, MessageCorrelation};
use crate::incoming::MessageSource;
use crate::ollama::{self, OllamaChatOptions, OllamaMessage};

// Conversation message format from web-agents
#[derive(Serialize, Deserialize, Debug)]
struct ConversationMessage {
    sender_id: usize,
    sender_name: String,
    receiver_id: usize,
    receiver_name: String,
    topic: String,
    message: String,
    timestamp: String,
    /// When present, overrides the `?auto_respond=` query flag for this request.
    #[serde(default)]
    auto_respond: Option<bool>,
}

// Evaluator result format from web-agents (Agent Evaluator)
#[derive(Serialize, Deserialize, Debug)]
struct EvaluatorResult {
    evaluator_name: String,
    sentiment: String,
    message: String,
    timestamp: String,
    #[serde(default)]
    auto_respond: Option<bool>,
}

/// OpenAI-compatible chat completion request (local Ollama backend only).
#[derive(Deserialize, Debug)]
struct OpenAiChatCompletionRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    stream: Option<bool>,
}

#[derive(Deserialize, Debug)]
struct OpenAiMessage {
    role: String,
    #[serde(default)]
    content: String,
}

#[derive(Serialize)]
struct OpenAiModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct OpenAiModelsList {
    object: &'static str,
    data: Vec<OpenAiModelObject>,
}

fn auto_respond_from_query(uri: &hyper::Uri) -> bool {
    uri.query().map_or(false, |q| {
        for pair in q.split('&') {
            let mut parts = pair.splitn(2, '=');
            let key = parts.next().unwrap_or("");
            if key == "auto_respond" {
                let v = parts.next().unwrap_or("");
                return v.eq_ignore_ascii_case("true") || v == "1";
            }
        }
        false
    })
}

/// Start the HTTP server that receives POST requests
pub async fn start_server(
    addr: SocketAddr,
    sender: mpsc::Sender<ChatMessage>,
    enabled: Arc<Mutex<bool>>,
    audit: Arc<audit::AuditHandle>,
    conversation_id: Arc<Mutex<String>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(addr = %addr, "HTTP server listening");

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let sender_clone = sender.clone();
        let enabled_clone = enabled.clone();
        let audit_clone = audit.clone();
        let conversation_id_shared = conversation_id.clone();

        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(
                    io,
                    service_fn(move |req| {
                        handle_request(
                            req,
                            sender_clone.clone(),
                            enabled_clone.clone(),
                            audit_clone.clone(),
                            conversation_id_shared.clone(),
                        )
                    }),
                )
                .await
            {
                tracing::warn!(?err, "error serving connection");
            }
        });
    }
}

async fn handle_request(
    req: Request<hyper::body::Incoming>,
    sender: mpsc::Sender<ChatMessage>,
    enabled: Arc<Mutex<bool>>,
    audit: Arc<audit::AuditHandle>,
    conversation_id_shared: Arc<Mutex<String>>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let is_enabled = *enabled.lock().unwrap();

    match (req.method(), req.uri().path()) {
        (&Method::POST, "/") => {
            let conversation_id = conversation_id_shared.lock().unwrap().clone();
            if !is_enabled {
                return Ok(Response::builder()
                    .status(StatusCode::SERVICE_UNAVAILABLE)
                    .body(Full::new(Bytes::from(
                        r#"{"status": "error", "message": "Server is disabled"}"#,
                    )))
                    .unwrap());
            }

            let query_auto_respond = auto_respond_from_query(req.uri());

            let request_id = audit::new_id();
            let event_id = audit::new_id();

            let body_bytes = match http_body_util::BodyExt::collect(req.into_body()).await {
                Ok(body) => body.to_bytes(),
                Err(_) => {
                    return Ok(Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .body(Full::new(Bytes::from("Failed to read request body")))
                        .unwrap());
                }
            };

            let body_str = String::from_utf8_lossy(&body_bytes);
            tracing::debug!(
                request_id = %request_id,
                body_len = body_bytes.len(),
                "received POST body"
            );

            let (message, payload_kind, audit_ts, api_auto_respond_effective) =
                match serde_json::from_str::<ConversationMessage>(&body_str) {
                    Ok(conv_msg) => {
                        let ts = audit::resolve_from_optional_payload(Some(conv_msg.timestamp.as_str()));
                        let api_auto =
                            conv_msg
                                .auto_respond
                                .unwrap_or(query_auto_respond);
                        tracing::info!(
                            request_id = %request_id,
                            sender = %conv_msg.sender_name,
                            auto_respond = api_auto,
                            "parsed conversation JSON"
                        );
                        (
                            ChatMessage {
                                content: conv_msg.message,
                                from: Some(conv_msg.sender_name),
                                correlation: Some(MessageCorrelation {
                                    conversation_id: conversation_id.clone(),
                                    event_id: event_id.clone(),
                                    request_id: request_id.clone(),
                                    timestamp_rfc3339: ts.clone(),
                                }),
                                source: MessageSource::Api,
                                api_auto_respond: api_auto,
                                assistant_generation: None,
                            },
                            "conversation",
                            ts,
                            api_auto,
                        )
                    }
                    Err(_) => match serde_json::from_str::<EvaluatorResult>(&body_str) {
                        Ok(eval_result) => {
                            let ts = audit::resolve_from_optional_payload(Some(eval_result.timestamp.as_str()));
                            let api_auto = eval_result
                                .auto_respond
                                .unwrap_or(query_auto_respond);
                            tracing::info!(
                                request_id = %request_id,
                                evaluator = %eval_result.evaluator_name,
                                sentiment = %eval_result.sentiment,
                                auto_respond = api_auto,
                                "parsed evaluator JSON"
                            );
                            (
                                ChatMessage {
                                    content: format!(
                                        "{}: {}",
                                        eval_result.sentiment, eval_result.message
                                    ),
                                    from: Some(eval_result.evaluator_name),
                                    correlation: Some(MessageCorrelation {
                                        conversation_id: conversation_id.clone(),
                                        event_id: event_id.clone(),
                                        request_id: request_id.clone(),
                                        timestamp_rfc3339: ts.clone(),
                                    }),
                                    source: MessageSource::Api,
                                    api_auto_respond: api_auto,
                                    assistant_generation: None,
                                },
                                "evaluator",
                                ts,
                                api_auto,
                            )
                        }
                        Err(_) => {
                            let ts = audit::resolve_from_optional_payload(None);
                            let api_auto = query_auto_respond;
                            (
                                ChatMessage {
                                    content: body_str.to_string(),
                                    from: Some("API".to_string()),
                                    correlation: Some(MessageCorrelation {
                                        conversation_id: conversation_id.clone(),
                                        event_id: event_id.clone(),
                                        request_id: request_id.clone(),
                                        timestamp_rfc3339: ts.clone(),
                                    }),
                                    source: MessageSource::Api,
                                    api_auto_respond: api_auto,
                                    assistant_generation: None,
                                },
                                "plain",
                                ts,
                                api_auto,
                            )
                        }
                    },
                };

            let record = AuditRecord {
                schema_version: SCHEMA_VERSION,
                kind: "http_in",
                ts: audit_ts,
                conversation_id: conversation_id.clone(),
                request_id: request_id.clone(),
                event_id: event_id.clone(),
                details: serde_json::json!({
                    "body_len": body_bytes.len(),
                    "payload_kind": payload_kind,
                    "auto_respond": api_auto_respond_effective,
                    "query_auto_respond": query_auto_respond,
                }),
            };
            if let Err(e) = audit.append_json_line(&record) {
                tracing::warn!(error = %e, request_id = %request_id, "failed to write audit line");
            }

            sender.send(message).ok();

            let body = serde_json::json!({
                "status": "ok",
                "message": "Message received",
                "request_id": request_id,
                "auto_respond": api_auto_respond_effective,
            })
            .to_string();

            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "application/json")
                .body(Full::new(Bytes::from(body)))
                .unwrap())
        }
        (&Method::POST, "/v1/chat/completions") => {
            let conversation_id = conversation_id_shared.lock().unwrap().clone();
            if !is_enabled {
                return Ok(json_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Server is disabled",
                ));
            }

            let body_bytes = match http_body_util::BodyExt::collect(req.into_body()).await {
                Ok(body) => body.to_bytes(),
                Err(_) => {
                    return Ok(json_error_response(
                        StatusCode::BAD_REQUEST,
                        "Failed to read request body",
                    ));
                }
            };

            let parsed: OpenAiChatCompletionRequest = match serde_json::from_slice(&body_bytes) {
                Ok(p) => p,
                Err(e) => {
                    return Ok(json_error_response(
                        StatusCode::BAD_REQUEST,
                        &format!("Invalid JSON: {e}"),
                    ));
                }
            };

            if parsed.stream == Some(true) {
                return Ok(json_error_response(
                    StatusCode::BAD_REQUEST,
                    "Streaming is not supported; send \"stream\": false or omit stream",
                ));
            }

            if parsed.messages.is_empty() {
                return Ok(json_error_response(
                    StatusCode::BAD_REQUEST,
                    "messages must be a non-empty array",
                ));
            }

            let request_id = audit::new_id();
            let event_id = audit::new_id();
            let openai_record = AuditRecord {
                schema_version: SCHEMA_VERSION,
                kind: "openai_chat_in",
                ts: audit::now_rfc3339(),
                conversation_id: conversation_id.clone(),
                request_id: request_id.clone(),
                event_id,
                details: serde_json::json!({
                    "model": parsed.model,
                    "messages_count": parsed.messages.len(),
                    "max_tokens": parsed.max_tokens,
                    "temperature": parsed.temperature,
                    "seed": parsed.seed,
                    "stream": false,
                    "endpoint": "/v1/chat/completions",
                }),
            };
            if let Err(e) = audit.append_json_line(&openai_record) {
                tracing::warn!(error = %e, request_id = %request_id, "openai audit append failed");
            }

            let model = parsed.model;
            let model_for_response = model.clone();
            let messages: Vec<OllamaMessage> = parsed
                .messages
                .into_iter()
                .map(|m| OllamaMessage {
                    role: m.role.to_lowercase(),
                    content: m.content,
                })
                .collect();
            let options = OllamaChatOptions {
                num_predict: parsed.max_tokens.map(|n| n as i32),
                temperature: parsed.temperature,
                seed: parsed.seed,
            };

            let audit_clone = audit.clone();
            let conv_for_task = conversation_id.clone();
            let rid = request_id.clone();
            let join = tokio::task::spawn_blocking(move || {
                ollama::chat_completion_with_audit(
                    &model,
                    &messages,
                    &options,
                    audit_clone,
                    conv_for_task,
                    rid,
                )
            })
            .await;

            match join {
                Ok(Ok(parsed_reply)) => {
                    let created = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let pt = parsed_reply.prompt_eval_count.unwrap_or(0);
                    let ct = parsed_reply.eval_count.unwrap_or(0);
                    let body = serde_json::json!({
                        "id": format!("chatcmpl-{request_id}"),
                        "object": "chat.completion",
                        "created": created,
                        "model": model_for_response,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": parsed_reply.content,
                            },
                            "finish_reason": "stop",
                        }],
                        "usage": {
                            "prompt_tokens": pt,
                            "completion_tokens": ct,
                            "total_tokens": pt.saturating_add(ct),
                        },
                    });
                    Ok(Response::builder()
                        .status(StatusCode::OK)
                        .header("Content-Type", "application/json")
                        .body(Full::new(Bytes::from(body.to_string())))
                        .unwrap())
                }
                Ok(Err(err_msg)) => Ok(json_error_response(
                    StatusCode::BAD_GATEWAY,
                    &format!("Ollama: {err_msg}"),
                )),
                Err(e) => Ok(json_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("completion task failed: {e}"),
                )),
            }
        }
        (&Method::GET, "/v1/models") => {
            if !is_enabled {
                return Ok(json_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Server is disabled",
                ));
            }
            let join = tokio::task::spawn_blocking(|| ollama::list_ollama_models_blocking()).await;
            match join {
                Ok(Ok(names)) => {
                    let data: Vec<OpenAiModelObject> = names
                        .into_iter()
                        .map(|id| OpenAiModelObject {
                            id,
                            object: "model",
                            created: 0,
                            owned_by: "ollama",
                        })
                        .collect();
                    let list = OpenAiModelsList {
                        object: "list",
                        data,
                    };
                    let body = serde_json::to_string(&list).unwrap_or_else(|_| "{}".into());
                    Ok(Response::builder()
                        .status(StatusCode::OK)
                        .header("Content-Type", "application/json")
                        .body(Full::new(Bytes::from(body)))
                        .unwrap())
                }
                Ok(Err(e)) => Ok(json_error_response(
                    StatusCode::BAD_GATEWAY,
                    &format!("Ollama: {e}"),
                )),
                Err(e) => Ok(json_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("models task failed: {e}"),
                )),
            }
        }
        (&Method::GET, "/health") => {
            if !is_enabled {
                return Ok(Response::builder()
                    .status(StatusCode::SERVICE_UNAVAILABLE)
                    .body(Full::new(Bytes::from("SERVICE_UNAVAILABLE")))
                    .unwrap());
            }
            Ok(Response::new(Full::new(Bytes::from("OK"))))
        }
        _ => Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::from("Not Found")))
            .unwrap()),
    }
}

fn json_error_response(status: StatusCode, message: &str) -> Response<Full<Bytes>> {
    let err_type = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    };
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": err_type,
            "param": null,
            "code": null,
        }
    });
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())))
        .unwrap()
}

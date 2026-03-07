use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use std::sync::mpsc;
use serde::{Deserialize, Serialize};

use crate::chat::ChatMessage;

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
}

/// Start the HTTP server that receives POST requests
pub async fn start_server(
    addr: SocketAddr,
    sender: mpsc::Sender<ChatMessage>,
    enabled: Arc<Mutex<bool>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = TcpListener::bind(addr).await?;
    println!("HTTP server listening on http://{}", addr);

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let sender_clone = sender.clone();
        let enabled_clone = enabled.clone();

        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(io, service_fn(move |req| handle_request(req, sender_clone.clone(), enabled_clone.clone())))
                .await
            {
                eprintln!("Error serving connection: {:?}", err);
            }
        });
    }
}

async fn handle_request(
    req: Request<hyper::body::Incoming>,
    sender: mpsc::Sender<ChatMessage>,
    enabled: Arc<Mutex<bool>>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    // Check if server is enabled
    let is_enabled = *enabled.lock().unwrap();
    
    match (req.method(), req.uri().path()) {
        (&Method::POST, "/") => {
            if !is_enabled {
                return Ok(Response::builder()
                    .status(StatusCode::SERVICE_UNAVAILABLE)
                    .body(Full::new(Bytes::from(r#"{"status": "error", "message": "Server is disabled"}"#)))
                    .unwrap());
            }
            
            // Read the request body
            let body_bytes = match http_body_util::BodyExt::collect(req.into_body()).await {
                Ok(body) => body.to_bytes(),
                Err(_) => {
                    return Ok(Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .body(Full::new(Bytes::from("Failed to read request body")))
                        .unwrap());
                }
            };

            // Try to parse as JSON first, then fall back to plain text
            let body_str = String::from_utf8_lossy(&body_bytes);
            println!("Received POST request: {}", body_str);

            // Try to parse as ConversationMessage JSON format
            let message = match serde_json::from_str::<ConversationMessage>(&body_str) {
                Ok(conv_msg) => {
                    // Successfully parsed JSON - use sender_name and message
                    println!("Parsed JSON message from: {}", conv_msg.sender_name);
                    ChatMessage {
                        content: conv_msg.message,
                        from: Some(conv_msg.sender_name),
                    }
                }
                Err(_) => {
                    // Not valid JSON or different format - treat as plain text
                    // Try to extract sender name if it's a simple format, otherwise use "API"
                    ChatMessage {
                        content: body_str.to_string(),
                        from: Some("API".to_string()),
                    }
                }
            };

            // Send message to the chat UI via sender
            sender.send(message).ok();

            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "application/json")
                .body(Full::new(Bytes::from(r#"{"status": "ok", "message": "Message received"}"#)))
                .unwrap())
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
        _ => {
            Ok(Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Full::new(Bytes::from("Not Found")))
                .unwrap())
        }
    }
}

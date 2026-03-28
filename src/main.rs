mod app;
mod audit;
mod chat;
mod incoming;
mod ollama;
mod server;

use std::net::SocketAddr;
use std::sync::{mpsc, Arc, Mutex};
use app::MyApp;
use chat::{ChatExample, ChatMessage};

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let conversation_id = audit::new_id();
    let audit_handle: Arc<audit::AuditHandle> = match audit::AuditHandle::open("openchat-audit.jsonl") {
        Ok(h) => Arc::new(h),
        Err(e) => {
            tracing::warn!(error = %e, "failed to open openchat-audit.jsonl; audit logging disabled");
            Arc::new(audit::AuditHandle::disabled())
        }
    };
    tracing::info!(conversation_id = %conversation_id, path = ?audit_handle.path(), "ams-chat startup");

    // Create a channel to bridge HTTP server and UI inbox
    let (tx, rx) = mpsc::channel::<ChatMessage>();
    
    // Create shared flag for server enable/disable
    let server_enabled = Arc::new(Mutex::new(true));
    
    // Create chat instance
    let chat = ChatExample::new();
    
    // Spawn a task to forward messages from HTTP server to UI inbox
    let inbox_sender = chat.inbox().sender();
    std::thread::spawn(move || {
        while let Ok(msg) = rx.recv() {
            inbox_sender.send(msg).ok();
        }
    });
    
    // Start HTTP server in background
    let server_tx = tx.clone();
    let server_enabled_clone = server_enabled.clone();
    let server_audit = audit_handle.clone();
    let server_conversation_id = conversation_id.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
        rt.block_on(async {
            if let Err(e) = server::start_server(
                addr,
                server_tx,
                server_enabled_clone,
                server_audit,
                server_conversation_id,
            )
            .await
            {
                tracing::error!(error = %e, "HTTP server error");
            }
        });
    });

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([900.0, 720.0])
            .with_min_inner_size([700.0, 700.0]) // Minimum width: 500px, minimum height: 400px
            .with_drag_and_drop(true),
        ..Default::default()
    };

    let server_enabled_for_app = server_enabled.clone();
    let app_audit = audit_handle.clone();
    let app_conversation_id = conversation_id.clone();
    eframe::run_native(
        "ams-chat",
        options,
        Box::new(move |_cc| {
            let mut app = MyApp::default();
            app.chat = chat;
            app.server_enabled = server_enabled_for_app;
            app.audit = app_audit;
            app.conversation_id = app_conversation_id;
            Ok(Box::new(app))
        }),
    )
}

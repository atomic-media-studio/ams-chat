mod app;
mod chat;
mod server;
mod ollama;

use std::net::SocketAddr;
use std::sync::{mpsc, Arc, Mutex};
use app::MyApp;
use chat::{ChatExample, ChatMessage};

fn main() -> eframe::Result<()> {
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
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
        rt.block_on(async {
            if let Err(e) = server::start_server(addr, server_tx, server_enabled_clone).await {
                eprintln!("Server error: {}", e);
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
    eframe::run_native(
        "ams-chat",
        options,
        Box::new(move |_cc| {
            let mut app = MyApp::default();
            app.chat = chat;
            app.server_enabled = server_enabled_for_app;
            Ok(Box::new(app))
        }),
    )
}

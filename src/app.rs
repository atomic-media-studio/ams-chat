use eframe::egui;
use egui::Frame;

use crate::audit::AuditHandle;
use crate::chat::ChatExample;
use crate::incoming::MessageSource;
use crate::ollama::{OllamaController, OllamaStatus};
use crate::store::{ConversationSettings, ConversationSummary, Store};

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct MyApp {
    pub chat: ChatExample,
    pub selected_model: String,
    /// Active conversation; shared with the HTTP server for inbound message attribution.
    pub conversation_id: Arc<Mutex<String>>,
    pub store: Arc<Store>,
    pub audit: Arc<AuditHandle>,
    conversation_list: Vec<ConversationSummary>,
    conversation_pick: String,
    last_saved_settings: Option<ConversationSettings>,
    server_status: ServerStatus,
    pub server_enabled: Arc<Mutex<bool>>,
    /// Set once on first frame so Catppuccin Latte replaces default dark styling.
    theme_applied: bool,
    ollama: OllamaController,
    pub selected_ollama_model: Arc<Mutex<String>>,
    ollama_input_text: String,
    ollama_token_limit: i32,
    ollama_token_limit_enabled: bool,
    chat_token_limit: i32,
    chat_token_limit_enabled: bool,
    chat_use_mode: ChatUseMode,
    download_chat_format: DownloadChatFormat,
    download_keyboard_format: DownloadChatFormat,
    keyboard_recording: bool,
    keyboard_input_log: Vec<(String, String)>,
    left_column_tab: LeftColumnTab,
}

#[derive(Clone, Copy, PartialEq)]
enum ServerStatus {
    Running,
    Stopped,
}

#[derive(Clone, Copy, PartialEq, Default)]
enum LeftColumnTab {
    #[default]
    General,
    About,
}

#[derive(Clone, Copy, PartialEq, Default)]
enum DownloadChatFormat {
    #[default]
    Json,
    Csv,
}

#[derive(Clone, Copy, PartialEq, Default)]
enum ChatUseMode {
    #[default]
    HumanAi,
    AiAi,
}

impl Default for MyApp {
    fn default() -> Self {
        let ollama = OllamaController::new();
        ollama.check_status(); // Check Ollama status on startup
        ollama.fetch_models(); // Fetch available models on startup

        Self {
            chat: ChatExample::default(),
            selected_model: String::new(),
            conversation_id: Arc::new(Mutex::new(String::new())),
            store: Arc::new(Store::open(":memory:").expect("in-memory store")),
            audit: Arc::new(crate::audit::AuditHandle::disabled()),
            conversation_list: Vec::new(),
            conversation_pick: String::new(),
            last_saved_settings: None,
            server_status: ServerStatus::Running, // Assume running since server starts before UI
            server_enabled: Arc::new(Mutex::new(true)),
            theme_applied: false,
            ollama,
            selected_ollama_model: Arc::new(Mutex::new(String::new())),
            ollama_input_text: String::new(),
            ollama_token_limit: 70,
            ollama_token_limit_enabled: false,
            chat_token_limit: 70,
            chat_token_limit_enabled: false,
            chat_use_mode: ChatUseMode::default(),
            download_chat_format: DownloadChatFormat::default(),
            download_keyboard_format: DownloadChatFormat::default(),
            keyboard_recording: false,
            keyboard_input_log: Vec::new(),
            left_column_tab: LeftColumnTab::default(),
        }
    }
}

impl MyApp {
    pub fn conv_id(&self) -> String {
        self.conversation_id.lock().unwrap().clone()
    }

    pub fn apply_loaded_settings(&mut self, s: ConversationSettings) {
        self.selected_model = s.selected_model.clone();
        *self.selected_ollama_model.lock().unwrap() = s.selected_model.clone();
        self.chat_token_limit = s.chat_token_limit;
        self.chat_token_limit_enabled = s.chat_token_limit_enabled;
        self.ollama_token_limit = s.ollama_token_limit;
        self.ollama_token_limit_enabled = s.ollama_token_limit_enabled;
        self.last_saved_settings = Some(s);
    }

    fn switch_to_conversation(&mut self, id: &str) {
        match self.store.load_messages(id) {
            Ok((msgs, ts)) => {
                *self.conversation_id.lock().unwrap() = id.to_string();
                self.conversation_pick = id.to_string();
                self.chat.hydrate(msgs, ts);
                if let Ok(s) = self.store.load_conversation_settings(id) {
                    self.apply_loaded_settings(s);
                }
            }
            Err(e) => tracing::warn!(error = %e, "load messages"),
        }
    }

    fn close_conversation_tab(&mut self, id: &str) {
        if let Err(e) = self.store.delete_conversation(id) {
            tracing::warn!(error = %e, "delete conversation");
            return;
        }
        if self.conv_id() != id {
            return;
        }
        match self.store.most_recent_conversation_id() {
            Ok(Some(new_id)) => {
                self.switch_to_conversation(&new_id);
            }
            Ok(None) => match self.store.create_conversation() {
                Ok(new_id) => {
                    *self.conversation_id.lock().unwrap() = new_id.clone();
                    self.conversation_pick = new_id.clone();
                    self.chat.reset_to_welcome();
                    let (w, t) = ChatExample::default_welcome();
                    let _ = self.store.append_message(&new_id, &w, &t);
                    if let Ok(s) = self.store.load_conversation_settings(&new_id) {
                        self.apply_loaded_settings(s);
                    }
                }
                Err(e) => tracing::warn!(error = %e, "create conversation after close"),
            },
            Err(e) => tracing::warn!(error = %e, "most_recent_conversation_id"),
        }
    }

    fn snapshot_settings(&self) -> ConversationSettings {
        ConversationSettings {
            selected_model: self.selected_model.clone(),
            chat_token_limit: self.chat_token_limit,
            chat_token_limit_enabled: self.chat_token_limit_enabled,
            ollama_token_limit: self.ollama_token_limit,
            ollama_token_limit_enabled: self.ollama_token_limit_enabled,
        }
    }

    fn current_timestamp_string() -> String {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let day_secs = now_secs % 86_400;
        let hours = day_secs / 3_600;
        let minutes = (day_secs % 3_600) / 60;
        let seconds = day_secs % 60;
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_applied {
            catppuccin_egui::set_theme(ctx, catppuccin_egui::LATTE);
            let mut fonts = egui::FontDefinitions::default();
            egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
            ctx.set_fonts(fonts);
            self.theme_applied = true;
        }

        self.conversation_list = self.store.list_conversations(50).unwrap_or_default();
        if self.conversation_pick.is_empty() {
            self.conversation_pick = self.conv_id();
        }

        let store_commit = self.store.clone();
        let conv_arc = self.conversation_id.clone();
        self.chat.set_message_commit_hook(Some(Box::new(move |msg, ts| {
            let id = conv_arc.lock().unwrap().clone();
            if let Err(e) = store_commit.append_message(&id, msg, ts) {
                tracing::warn!(error = %e, "sqlite persist failed");
            }
        })));

        let snap = self.snapshot_settings();
        if self.last_saved_settings.as_ref() != Some(&snap) {
            if let Err(e) = self.store.save_conversation_settings(&self.conv_id(), &snap) {
                tracing::warn!(error = %e, "sqlite settings save failed");
            }
            self.last_saved_settings = Some(snap);
        }

        if self.keyboard_recording {
            let events = ctx.input(|i| i.events.clone());
            for event in events {
                match event {
                    egui::Event::Text(text) if !text.is_empty() => {
                        self.keyboard_input_log
                            .push((Self::current_timestamp_string(), text));
                    }
                    egui::Event::Key {
                        key,
                        pressed: true,
                        modifiers,
                        ..
                    } => {
                        let mut key_repr = String::new();
                        if modifiers.ctrl {
                            key_repr.push_str("Ctrl+");
                        }
                        if modifiers.alt {
                            key_repr.push_str("Alt+");
                        }
                        if modifiers.shift {
                            key_repr.push_str("Shift+");
                        }
                        if modifiers.mac_cmd || modifiers.command {
                            key_repr.push_str("Cmd+");
                        }
                        key_repr.push_str(&format!("{key:?}"));
                        self.keyboard_input_log
                            .push((Self::current_timestamp_string(), key_repr));
                    }
                    _ => {}
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let panel_gap = 8.0;
            let available_width = ui.available_width();
            let left_width = 200.0; // Fixed width of 200px
            let _right_width = 0.0;
            let available_height = ui.available_height();
            let horizontal_spacing = ui.spacing().item_spacing.x;
            // Slightly narrower chat column to avoid horizontal overflow (tunable).
            let center_width = (available_width
                - (panel_gap * 2.0)
                - left_width
                - horizontal_spacing
                - 16.0)
                .max(0.0);
                
            let content_height = (available_height - (panel_gap * 2.0)).max(0.0);

            let panel_fill = ui.visuals().panel_fill;
            let panel_border = egui::Stroke::new(
                1.0,
                ui.visuals().widgets.noninteractive.bg_stroke.color,
            );
            let left_inner_margin = 8_i8;
            
            ui.add_space(panel_gap);
            ui.horizontal(|ui| {
                // Left column: fixed width — does not shrink/grow when switching General vs About.
                ui.allocate_ui_with_layout(
                    egui::vec2(left_width, content_height),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        Frame::default()
                            .fill(panel_fill)
                            .stroke(panel_border)
                            .corner_radius(4.0)
                            .inner_margin(egui::Margin::same(left_inner_margin))
                            .outer_margin(0.0)
                            .show(ui, |ui| {
                                let col_w = ui.available_width();
                                ui.set_min_width(col_w);
                                ui.set_max_width(col_w);
                                ui.vertical(|ui| {
                            // Top bar with tabs
                            ui.horizontal(|ui| {
                                let general_selected = self.left_column_tab == LeftColumnTab::General;
                                let about_selected = self.left_column_tab == LeftColumnTab::About;
                                if ui.selectable_label(general_selected, "General").clicked() {
                                    self.left_column_tab = LeftColumnTab::General;
                                }
                                if ui.selectable_label(about_selected, "About").clicked() {
                                    self.left_column_tab = LeftColumnTab::About;
                                }
                            });
                            ui.add_space(4.0);
                            ui.separator();
                            ui.add_space(4.0);

                            match self.left_column_tab {
                                LeftColumnTab::General => {
                                    egui::CollapsingHeader::new("Chat Settings")
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                if ui
                                                    .selectable_label(
                                                        self.chat_use_mode == ChatUseMode::HumanAi,
                                                        "Human-Agent",
                                                    )
                                                    .clicked()
                                                {
                                                    self.chat_use_mode = ChatUseMode::HumanAi;
                                                    println!("Selected mode: Human-Agent");
                                                }
                                                if ui
                                                    .selectable_label(
                                                        self.chat_use_mode == ChatUseMode::AiAi,
                                                        "Agent-Agent",
                                                    )
                                                    .clicked()
                                                {
                                                    self.chat_use_mode = ChatUseMode::AiAi;
                                                    println!("Selected mode: Agent-Agent");
                                                }
                                            });
                                            ui.add_space(4.0);
                                            let ollama_models = self.ollama.models();
                                            let ollama_status_chat = self.ollama.status();
                                            if ollama_status_chat == OllamaStatus::Running && !ollama_models.is_empty() {
                                                egui::ComboBox::from_id_salt("chat_model_selector")
                                                    .selected_text(if self.selected_model.is_empty() {
                                                        "Select model"
                                                    } else {
                                                        &self.selected_model
                                                    })
                                                    .show_ui(ui, |ui| {
                                                        for model in &ollama_models {
                                                            if ui.selectable_label(self.selected_model == *model, model).clicked() {
                                                                self.selected_model = model.clone();
                                                            }
                                                        }
                                                    });
                                            } else {
                                                ui.label(egui::RichText::new("No models available").small().weak());
                                            }
                                            ui.add_space(4.0);
                                            ui.horizontal(|ui| {
                                                ui.checkbox(&mut self.chat_token_limit_enabled, "Token Limit");
                                                if self.chat_token_limit_enabled {
                                                    ui.label("Tokens:");
                                                    ui.add_sized(
                                                        [40.0, 18.0],
                                                        egui::DragValue::new(&mut self.chat_token_limit)
                                                            .range(1..=1000)
                                                            .speed(1.0),
                                                    );
                                                }
                                            });
                                            ui.add_space(4.0);
                                            ui.horizontal(|ui| {
                                                let label = if self.keyboard_recording {
                                                    "Stop Recording"
                                                } else {
                                                    "Record Keyboard"
                                                };
                                                if ui.button(label).clicked() {
                                                    self.keyboard_recording = !self.keyboard_recording;
                                                    if self.keyboard_recording {
                                                        self.keyboard_input_log.clear();
                                                    }
                                                }
                                                if self.keyboard_recording {
                                                    ui.label(egui::RichText::new("Recording").small().weak());
                                                }
                                            });
                                            ui.add_space(4.0);

                                            egui::CollapsingHeader::new("Download Chat Messages")
                                                .default_open(true)
                                                .show(ui, |ui| {
                                                    ui.horizontal(|ui| {
                                                        egui::ComboBox::from_id_salt("download_chat_format")
                                                            .selected_text(match self.download_chat_format {
                                                                DownloadChatFormat::Json => "JSON",
                                                                DownloadChatFormat::Csv => "CSV",
                                                            })
                                                            .show_ui(ui, |ui| {
                                                                ui.selectable_value(
                                                                    &mut self.download_chat_format,
                                                                    DownloadChatFormat::Json,
                                                                    "JSON",
                                                                );
                                                                ui.selectable_value(
                                                                    &mut self.download_chat_format,
                                                                    DownloadChatFormat::Csv,
                                                                    "CSV",
                                                                );
                                                            });

                                                        if ui.button("Download").clicked() {
                                                            let rows = self.chat.export_rows();
                                                            let (content, default_name) = match self.download_chat_format {
                                                                DownloadChatFormat::Json => {
                                                                    let data: Vec<serde_json::Value> = rows
                                                                        .into_iter()
                                                                        .map(|(timestamp, from, content)| {
                                                                            serde_json::json!({
                                                                                "timestamp": timestamp,
                                                                                "from": from,
                                                                                "content": content
                                                                            })
                                                                        })
                                                                        .collect();
                                                                    (
                                                                        serde_json::to_string_pretty(&data)
                                                                            .unwrap_or_else(|_| "[]".to_string()),
                                                                        "chat-export.json",
                                                                    )
                                                                }
                                                                DownloadChatFormat::Csv => {
                                                                    let mut csv =
                                                                        String::from("timestamp,from,content\n");
                                                                    for (timestamp, from, content) in rows {
                                                                        let esc =
                                                                            |s: &str| format!("\"{}\"", s.replace('\"', "\"\""));
                                                                        csv.push_str(
                                                                            &format!(
                                                                                "{},{},{}\n",
                                                                                esc(&timestamp),
                                                                                esc(&from),
                                                                                esc(&content)
                                                                            ),
                                                                        );
                                                                    }
                                                                    (csv, "chat-export.csv")
                                                                }
                                                            };

                                                            if let Some(path) = rfd::FileDialog::new()
                                                                .set_file_name(default_name)
                                                                .save_file()
                                                            {
                                                                if let Err(err) = std::fs::write(path, content) {
                                                                    eprintln!("Failed to save chat export: {err}");
                                                                }
                                                            }
                                                        }
                                                    });
                                                });

                                            ui.add_space(4.0);

                                            egui::CollapsingHeader::new("Download Keyboard Input")
                                                .default_open(true)
                                                .show(ui, |ui| {
                                                    ui.horizontal(|ui| {
                                                        egui::ComboBox::from_id_salt("download_keyboard_format")
                                                            .selected_text(match self.download_keyboard_format {
                                                                DownloadChatFormat::Json => "JSON",
                                                                DownloadChatFormat::Csv => "CSV",
                                                            })
                                                            .show_ui(ui, |ui| {
                                                                ui.selectable_value(
                                                                    &mut self.download_keyboard_format,
                                                                    DownloadChatFormat::Json,
                                                                    "JSON",
                                                                );
                                                                ui.selectable_value(
                                                                    &mut self.download_keyboard_format,
                                                                    DownloadChatFormat::Csv,
                                                                    "CSV",
                                                                );
                                                            });

                                                        if ui.button("Download").clicked() {
                                                            let (content, default_name) = match self.download_keyboard_format {
                                                                DownloadChatFormat::Json => {
                                                                    let data: Vec<serde_json::Value> = self
                                                                        .keyboard_input_log
                                                                        .iter()
                                                                        .map(|(timestamp, stroke)| {
                                                                            serde_json::json!({
                                                                                "timestamp": timestamp,
                                                                                "stroke": stroke
                                                                            })
                                                                        })
                                                                        .collect();
                                                                    (
                                                                        serde_json::to_string_pretty(&data)
                                                                            .unwrap_or_else(|_| "[]".to_string()),
                                                                        "keyboard-input-export.json",
                                                                    )
                                                                }
                                                                DownloadChatFormat::Csv => {
                                                                    let mut csv = String::from("timestamp,stroke\n");
                                                                    for (timestamp, stroke) in &self.keyboard_input_log {
                                                                        let esc =
                                                                            |s: &str| format!("\"{}\"", s.replace('\"', "\"\""));
                                                                        csv.push_str(
                                                                            &format!("{},{}\n", esc(timestamp), esc(stroke)),
                                                                        );
                                                                    }
                                                                    (csv, "keyboard-input-export.csv")
                                                                }
                                                            };

                                                            if let Some(path) = rfd::FileDialog::new()
                                                                .set_file_name(default_name)
                                                                .save_file()
                                                            {
                                                                if let Err(err) = std::fs::write(path, content) {
                                                                    eprintln!(
                                                                        "Failed to save keyboard input export: {err}"
                                                                    );
                                                                }
                                                            }
                                                        }
                                                    });
                                                });
                                            ui.add_space(4.0);
                                            ui.label(egui::RichText::new("Conversations").small().strong());
                                            ui.horizontal(|ui| {
                                                if ui.button("New").clicked() {
                                                    match self.store.create_conversation() {
                                                        Ok(id) => {
                                                            *self.conversation_id.lock().unwrap() = id.clone();
                                                            self.conversation_pick = id.clone();
                                                            self.chat.reset_to_welcome();
                                                            let (w, t) = ChatExample::default_welcome();
                                                            let _ = self.store.append_message(&id, &w, &t);
                                                            if let Ok(s) = self.store.load_conversation_settings(&id) {
                                                                self.apply_loaded_settings(s);
                                                            }
                                                        }
                                                        Err(e) => tracing::warn!(error = %e, "create conversation"),
                                                    }
                                                }
                                                egui::ComboBox::from_id_salt("conv_pick")
                                                    .width(120.0)
                                                    .selected_text(if self.conversation_pick.len() > 8 {
                                                        format!("{}…", &self.conversation_pick[..8])
                                                    } else {
                                                        self.conversation_pick.clone()
                                                    })
                                                    .show_ui(ui, |ui| {
                                                        let conv_list = self.conversation_list.clone();
                                                        for c in &conv_list {
                                                            let label = if c.id.len() > 8 {
                                                                format!(
                                                                    "{}… · {}",
                                                                    &c.id[..8],
                                                                    &c.updated_at
                                                                )
                                                            } else {
                                                                format!("{} · {}", c.id, c.updated_at)
                                                            };
                                                            if ui
                                                                .selectable_label(self.conv_id() == c.id, label)
                                                                .clicked()
                                                            {
                                                                self.switch_to_conversation(&c.id);
                                                            }
                                                        }
                                                    });
                                            });
                                            ui.horizontal(|ui| {
                                                if ui.button("Export JSON…").clicked() {
                                                    match self.store.export_conversation_json(&self.conv_id()) {
                                                        Ok(json) => {
                                                            if let Some(path) = rfd::FileDialog::new()
                                                                .set_file_name("conversation.json")
                                                                .save_file()
                                                            {
                                                                if let Err(err) = std::fs::write(path, json) {
                                                                    tracing::warn!(error = %err, "export write");
                                                                }
                                                            }
                                                        }
                                                        Err(e) => tracing::warn!(error = %e, "export"),
                                                    }
                                                }
                                                if ui.button("Import JSON…").clicked() {
                                                    if let Some(path) = rfd::FileDialog::new().pick_file() {
                                                        match std::fs::read_to_string(path) {
                                                            Ok(s) => {
                                                                match self.store.import_conversation_json(&s) {
                                                                    Ok(new_id) => {
                                                                        *self.conversation_id.lock().unwrap() =
                                                                            new_id.clone();
                                                                        self.conversation_pick = new_id.clone();
                                                                        match self.store.load_messages(&new_id) {
                                                                            Ok((m, t)) => {
                                                                                self.chat.hydrate(m, t);
                                                                            }
                                                                            Err(e) => tracing::warn!(error = %e, "load after import"),
                                                                        }
                                                                    }
                                                                    Err(e) => tracing::warn!(error = %e, "import"),
                                                                }
                                                            }
                                                            Err(e) => tracing::warn!(error = %e, "read import file"),
                                                        }
                                                    }
                                                }
                                            });
                                            if ui.button("Clear chat").clicked() {
                                                let id = self.conv_id();
                                                let _ = self.store.delete_messages_for_conversation(&id);
                                                self.chat.reset_to_welcome();
                                                let (w, t) = ChatExample::default_welcome();
                                                let _ = self.store.append_message(&id, &w, &t);
                                            }
                                        });

                                    ui.add_space(8.0);
                                    ui.separator();
                                    ui.add_space(8.0);

                                    egui::CollapsingHeader::new("HTTP Server Status")
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            let (status_text, status_color) = match self.server_status {
                                                ServerStatus::Running => {
                                                    ("● Running", ui.visuals().strong_text_color())
                                                }
                                                ServerStatus::Stopped => {
                                                    ("● Stopped", ui.visuals().weak_text_color())
                                                }
                                            };

                                            ui.label(egui::RichText::new(status_text).color(status_color));
                                            ui.add_space(4.0);

                                            let is_enabled = *self.server_enabled.lock().unwrap();
                                            let button_text = if is_enabled { "OFF" } else { "ON" };

                                            if ui.button(button_text).clicked() {
                                                let mut enabled = self.server_enabled.lock().unwrap();
                                                *enabled = !*enabled;
                                                self.server_status = if *enabled {
                                                    ServerStatus::Running
                                                } else {
                                                    ServerStatus::Stopped
                                                };
                                            }

                                            ui.add_space(2.0);
                                            ui.label(egui::RichText::new("http://127.0.0.1:3000").small().weak());
                                        });

                                    ui.add_space(8.0);
                                    ui.separator();
                                    ui.add_space(8.0);

                                    let ollama_status = self.ollama.status();
                                    let models = self.ollama.models();
                                    let current_model = self.selected_ollama_model.lock().unwrap().clone();

                                    egui::CollapsingHeader::new("Ollama Status")
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            let (ollama_status_text, ollama_status_color) = match ollama_status {
                                                OllamaStatus::Running => {
                                                    ("● Running", ui.visuals().strong_text_color())
                                                }
                                                OllamaStatus::Stopped => {
                                                    ("● Stopped", ui.visuals().weak_text_color())
                                                }
                                                OllamaStatus::Checking => {
                                                    ("● Checking", ui.visuals().weak_text_color())
                                                }
                                            };

                                            ui.label(egui::RichText::new(ollama_status_text).color(ollama_status_color));
                                            ui.add_space(4.0);

                                            if ui.button("Check").clicked() {
                                                self.ollama.check_status();
                                                self.ollama.fetch_models();
                                            }

                                            ui.add_space(2.0);
                                            ui.label(egui::RichText::new("http://127.0.0.1:11434").small().weak());
                                        });

                                    ui.add_space(8.0);
                                    ui.separator();
                                    ui.add_space(8.0);

                                    egui::CollapsingHeader::new("Test Ollama API")
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            if ollama_status == OllamaStatus::Running && !models.is_empty() {
                                                egui::ComboBox::from_id_salt("ollama_model_selector")
                                                    .selected_text(if current_model.is_empty() {
                                                        "Select model"
                                                    } else {
                                                        &current_model
                                                    })
                                                    .show_ui(ui, |ui| {
                                                        for model in &models {
                                                            if ui.selectable_label(current_model == *model, model).clicked() {
                                                                *self.selected_ollama_model.lock().unwrap() = model.clone();
                                                            }
                                                        }
                                                    });
                                            } else if ollama_status == OllamaStatus::Stopped {
                                                ui.label(egui::RichText::new("Not installed").small().weak());
                                            } else if ollama_status == OllamaStatus::Checking {
                                                ui.label(egui::RichText::new("Checking...").small().weak());
                                            } else {
                                                ui.label(egui::RichText::new("No models available").small().weak());
                                            }
                                            ui.add_space(4.0);

                                            ui.horizontal(|ui| {
                                                ui.checkbox(&mut self.ollama_token_limit_enabled, "Token Limit");
                                                if self.ollama_token_limit_enabled {
                                                    ui.label("Tokens:");
                                                    ui.add_sized(
                                                        [40.0, 18.0],
                                                        egui::DragValue::new(&mut self.ollama_token_limit)
                                                            .range(1..=1000)
                                                            .speed(1.0),
                                                    );
                                                }
                                            });
                                            ui.add_space(4.0);

                                            ui.horizontal(|ui| {
                                                let total_width = ui.available_width();
                                                let input_width = total_width * 0.4;
                                                let widget_height = 20.0;
                                                let response = ui.add_sized(
                                                    [input_width, widget_height],
                                                    egui::TextEdit::singleline(&mut self.ollama_input_text),
                                                );
                                                let can_send = ollama_status == OllamaStatus::Running && !current_model.is_empty();
                                                let enter_pressed = response.lost_focus()
                                                    && ui.input(|i| i.key_pressed(egui::Key::Enter));
                                                let send_clicked = ui.button("Send").clicked();
                                                if (enter_pressed || send_clicked)
                                                    && can_send
                                                    && !self.ollama_input_text.trim().is_empty()
                                                {
                                                    let model = current_model.clone();
                                                    let message = self.ollama_input_text.clone();
                                                    let token_limit = if self.ollama_token_limit_enabled {
                                                        Some(self.ollama_token_limit)
                                                    } else {
                                                        None
                                                    };
                                                    let tx = self.chat.inbox().sender();
                                                    let system_message = crate::chat::ChatMessage {
                                                        content: format!("Testing Ollama API: {}", message),
                                                        from: Some("System".to_string()),
                                                        correlation: None,
                                                        source: MessageSource::System,
                                                        api_auto_respond: false,
                                                        assistant_generation: None,
                                                    };
                                                    tx.send(system_message).ok();
                                                    let request_id = crate::audit::new_id();
                                                    self.ollama.send_message(
                                                        model,
                                                        message,
                                                        token_limit,
                                                        self.audit.clone(),
                                                        self.conv_id(),
                                                        request_id,
                                                        Box::new(move |msg| {
                                                            tx.send(msg).ok();
                                                        }),
                                                    );
                                                    self.ollama_input_text.clear();
                                                }
                                            });
                                        });

                                }
                                LeftColumnTab::About => {
                                    ui.vertical(|ui| {
                                        ui.label("ams-chat");
                                    });
                                }
                            }
                                ui.add_space(0.0);
                            });
                            });
                    },
                );

                // Center column - 60% width: conversation tabs (intrinsic height) + chat (remainder)
                ui.allocate_ui_with_layout(
                    egui::vec2(center_width, content_height),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                    ui.set_min_height(content_height);
                    ui.set_width(center_width);

                    Frame::default()
                        .fill(panel_fill)
                        .stroke(panel_border)
                        .corner_radius(4.0)
                        .inner_margin(egui::Margin::symmetric(6, 4))
                        .outer_margin(0.0)
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.set_max_height(36.0);
                            egui::ScrollArea::horizontal()
                                .max_height(32.0)
                                .auto_shrink([true, true])
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 6.0;
                                        let conv_list = self.conversation_list.clone();
                                        if conv_list.is_empty() {
                                            ui.label(
                                                egui::RichText::new("No chats — create one in the sidebar")
                                                    .small()
                                                    .weak(),
                                            );
                                        } else {
                                            for c in &conv_list {
                                                let short = if c.id.len() > 12 {
                                                    format!("{}…", &c.id[..12])
                                                } else {
                                                    c.id.clone()
                                                };
                                                let selected = self.conv_id() == c.id;
                                                let cid = c.id.clone();
                                                ui.push_id(&cid, |ui| {
                                                    ui.horizontal(|ui| {
                                                        ui.spacing_mut().item_spacing.x = 2.0;
                                                        if ui.selectable_label(selected, short).clicked() {
                                                            self.switch_to_conversation(&cid);
                                                        }
                                                        if ui
                                                            .small_button(egui_phosphor::regular::X)
                                                            .on_hover_text("Close conversation")
                                                            .clicked()
                                                        {
                                                            self.close_conversation_tab(&cid);
                                                        }
                                                    });
                                                });
                                            }
                                        }
                                    });
                                });
                        });

                    ui.add_space(4.0);

                    let chat_body_height = ui.available_height().max(0.0);
                    Frame::default()
                        .fill(panel_fill)
                        .stroke(panel_border)
                        .corner_radius(4.0)
                        .inner_margin(0.0)
                        .outer_margin(0.0)
                        .show(ui, |ui| {
                            ui.set_min_width(center_width);
                            ui.set_max_width(center_width);
                            ui.set_height(chat_body_height);
                            
                            // Set up message handler for chat with current values
                            // Update each frame to ensure we have the latest model selection and settings
                            let selected_model = self.selected_model.clone();
                            let ollama_status = self.ollama.status();
                            let ollama_controller = self.ollama.clone();
                            let chat_token_limit = if self.chat_token_limit_enabled {
                                Some(self.chat_token_limit)
                            } else {
                                None
                            };
                            let tx = self.chat.inbox().sender();
                            
                            let waiting_flag = self.chat.waiting_for_response().clone();
                            let audit_for_chat = self.audit.clone();
                            let conversation_for_chat = self.conversation_id.clone();
                            self.chat.set_message_handler(Box::new(move |message: String| {
                                let tx_clone = tx.clone();
                                if selected_model.is_empty() {
                                    // No model selected, prompt user to select one
                                    let bot_message = crate::chat::ChatMessage {
                                        content: "Please select Ollama Model.".to_string(),
                                        from: Some("System".to_string()),
                                        correlation: None,
                                        source: MessageSource::System,
                                        api_auto_respond: false,
                                        assistant_generation: None,
                                    };
                                    tx_clone.send(bot_message).ok();
                                } else if ollama_status == crate::ollama::OllamaStatus::Running {
                                    // Model selected and Ollama is running, send to Ollama
                                    // Set waiting flag to true
                                    *waiting_flag.lock().unwrap() = true;
                                    
                                    let model_clone = selected_model.clone();
                                    let tx_for_ollama = tx_clone.clone();
                                    let waiting_flag_clone = waiting_flag.clone();
                                    let request_id = crate::audit::new_id();
                                    ollama_controller.send_message(
                                        model_clone,
                                        message,
                                        chat_token_limit,
                                        audit_for_chat.clone(),
                                        conversation_for_chat.lock().unwrap().clone(),
                                        request_id,
                                        Box::new(move |msg| {
                                            // Clear waiting flag when response arrives
                                            *waiting_flag_clone.lock().unwrap() = false;
                                            tx_for_ollama.send(msg).ok();
                                        }),
                                    );
                                } else {
                                    // Ollama not running
                                    let bot_message = crate::chat::ChatMessage {
                                        content: "Ollama is not running. Please check Ollama status.".to_string(),
                                        from: Some("System".to_string()),
                                        correlation: None,
                                        source: MessageSource::System,
                                        api_auto_respond: false,
                                        assistant_generation: None,
                                    };
                                    tx_clone.send(bot_message).ok();
                                }
                            }));

                            self.chat
                                .set_main_input_enabled(self.chat_use_mode == ChatUseMode::HumanAi);
                            self.chat.ui(ui);
                        });
                });
                
            });
            ui.add_space(panel_gap);
        });
    }
}




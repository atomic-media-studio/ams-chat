use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use egui::{Align, Frame, Layout, ScrollArea, Ui, Vec2};
use egui_inbox::UiInbox;

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub content: String,
    pub from: Option<String>,
}

#[derive(Debug)]
struct HistoryLoader {
    // Placeholder struct for future history loading functionality
}

impl HistoryLoader {
    pub fn new() -> Self {
        Self {}
    }
}

pub type MessageHandler = Box<dyn Fn(String) + Send + Sync>;

pub struct ChatExample {
    messages: Vec<ChatMessage>, // Simple Vec instead of InfiniteScroll
    message_timestamps: Vec<String>,
    inbox: UiInbox<ChatMessage>,
    #[allow(dead_code)]
    history_loader: Arc<HistoryLoader>,
    input_text: String,
    message_handler: Option<MessageHandler>,
    waiting_for_response: Arc<std::sync::Mutex<bool>>,
    picked_file_path: Option<String>,
}

impl Default for ChatExample {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatExample {
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

    pub fn new() -> Self {
        let history_loader = Arc::new(HistoryLoader::new());
        let inbox = UiInbox::new();
        
        // Load initial message from history
        let initial_messages = vec![
            ChatMessage {
                content: "ams-chat Started".to_string(),
                from: Some("System".to_string()),
            }
        ];

        ChatExample {
            messages: initial_messages,
            message_timestamps: vec![Self::current_timestamp_string()],
            inbox,
            history_loader,
            input_text: String::new(),
            message_handler: None,
            waiting_for_response: Arc::new(std::sync::Mutex::new(false)),
            picked_file_path: None,
        }
    }

    /// Get a reference to the inbox for external message injection (e.g., from HTTP server)
    pub fn inbox(&self) -> &UiInbox<ChatMessage> {
        &self.inbox
    }

    /// Set the message handler that will be called when a user sends a message
    pub fn set_message_handler(&mut self, handler: MessageHandler) {
        self.message_handler = Some(handler);
    }

    /// Get a reference to the waiting_for_response flag
    pub fn waiting_for_response(&self) -> &Arc<std::sync::Mutex<bool>> {
        &self.waiting_for_response
    }

    pub fn export_rows(&self) -> Vec<(String, String, String)> {
        self.messages
            .iter()
            .enumerate()
            .map(|(idx, msg)| {
                let ts = self
                    .message_timestamps
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| "--:--:--".to_string());
                let from = msg.from.clone().unwrap_or_else(|| "Unknown".to_string());
                (ts, from, msg.content.clone())
            })
            .collect()
    }

    pub fn clear_messages(&mut self) {
        self.messages.clear();
        self.message_timestamps.clear();
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        // Read incoming messages from inbox
        self.inbox.read(ui).for_each(|message| {
            // Only add non-empty messages to prevent spacing issues
            if !message.content.trim().is_empty() {
                self.messages.push(message);
                self.message_timestamps.push(Self::current_timestamp_string());
            }
        });

        // Use all available height, with input panel at bottom
        ui.vertical(|ui| {

            // Chat messages area - takes remaining space
            let available_height = ui.available_height();

            // Calculate space needed for input panel
            let input_upward_spacing = 0.0; // How much the input is moved up from bottom

            let input_height = 26.0; // Height of input controls

            let input_margin = 4.0; // Additional margin/spacing

            let extra_scroll_padding = 80.0; // Extra padding to prevent scroll area from going too far down

            let input_panel_height = input_upward_spacing + input_height + input_margin + extra_scroll_padding;
            let top_padding = 22.0; // 12.0 + 10.0 to move pink rectangle 10px down
            let messages_area_height = (available_height - input_panel_height - top_padding - 20.0).max(0.0);

            Frame::none()
                .inner_margin(egui::Margin { left: 0.0, right: 0.0, top: top_padding, bottom: 0.0 })
                .show(ui, |ui| {
                    ScrollArea::vertical()
                        .animated(false)
                        .auto_shrink([false, false])
                        .stick_to_bottom(true)
                        .max_height(messages_area_height)
                        .show(ui, |ui| {
                            let row_width = ui.available_width();
                            let left_margin = 10.0;
                            let right_margin = 10.0;

                            for (idx, item) in self.messages.iter().enumerate() {
                                let timestamp = self
                                    .message_timestamps
                                    .get(idx)
                                    .map(|s| s.as_str())
                                    .unwrap_or("--:--:--");

                                ui.allocate_ui_with_layout(
                                    egui::vec2(row_width, 0.0),
                                    Layout::left_to_right(Align::Min),
                                    |ui| {
                                    ui.spacing_mut().item_spacing.x = 8.0;
                                    let separator_color = ui.visuals().widgets.noninteractive.bg_stroke.color;
                                    Frame::default()
                                        .fill(egui::Color32::TRANSPARENT)
                                        .stroke(egui::Stroke::new(1.0, separator_color))
                                        .rounding(4.0)
                                        .inner_margin(egui::Margin::same(6.0))
                                        .outer_margin(egui::Margin {
                                            left: 6.0,
                                            right: 0.0,
                                            top: 0.0,
                                            bottom: 4.0,
                                        })
                                        .show(ui, |ui| {
                                            ui.label(
                                                egui::RichText::new(timestamp)
                                                    .size(12.0)
                                                    .color(egui::Color32::WHITE),
                                            );
                                        });
                                    ui.separator();

                                    let messages_middle_width = ui.available_width().max(0.0);
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(messages_middle_width, 0.0),
                                        Layout::top_down(Align::Min),
                                        |ui| {
                                            let max_msg_width =
                                                (messages_middle_width - left_margin - right_margin).max(0.0);
                                            ui.set_min_width(messages_middle_width);
                                            ui.set_max_width(messages_middle_width);
                                            let layout = Layout::top_down(Align::Min);
                                            ui.with_layout(layout, |ui| {
                                                ui.set_max_width(max_msg_width);
                                                let msg_color = egui::Color32::from_rgb(50, 50, 50);
                                                let border_color = match item.from.as_deref() {
                                                    Some("Human") => egui::Color32::from_rgb(0, 255, 0),
                                                    Some(from) if from.starts_with("Ollama") => {
                                                        egui::Color32::from_rgb(255, 255, 0)
                                                    }
                                                    Some("Agent Evaluator") => egui::Color32::from_rgb(255, 105, 180),
                                                    Some("Agent Manager") => egui::Color32::from_rgb(255, 0, 0),
                                                    Some(from) if from.starts_with("Agent") => {
                                                        egui::Color32::from_rgb(255, 255, 0)
                                                    }
                                                    Some("System") | Some("API") => {
                                                        egui::Color32::from_rgb(204, 85, 0)
                                                    }
                                                    _ => egui::Color32::TRANSPARENT,
                                                };
                                                let border_width = if border_color != egui::Color32::TRANSPARENT {
                                                    1.0
                                                } else {
                                                    0.0
                                                };
                                                let rounding = 4.0;
                                                let margin = 8.0;
                                                let outer_margin = egui::Margin {
                                                    left: left_margin,
                                                    right: right_margin,
                                                    top: 0.0,
                                                    bottom: 4.0,
                                                };
                                                let content_max_width = max_msg_width - margin * 2.0;

                                                if item.from.as_deref() == Some("Agent Manager") {
                                                    ui.add_space(4.0);
                                                }

                                                Frame::default()
                                                    .inner_margin(margin)
                                                    .outer_margin(outer_margin)
                                                    .fill(msg_color)
                                                    .rounding(rounding)
                                                    .stroke(egui::Stroke::new(border_width, border_color))
                                                    .show(ui, |ui| {
                                                        ui.set_max_width(content_max_width);
                                                        ui.with_layout(Layout::top_down(Align::Min), |ui| {
                                                            let header_color = egui::Color32::WHITE;
                                                            let content_color =
                                                                egui::Color32::from_rgba_unmultiplied(150, 150, 150, 255);

                                                            if let Some(from) = &item.from {
                                                                if from.starts_with("Ollama ") {
                                                                    let parts: Vec<&str> = from.splitn(2, ' ').collect();
                                                                    if parts.len() == 2 {
                                                                        ui.horizontal(|ui| {
                                                                            ui.label(egui::RichText::new("Ollama").strong().color(header_color));
                                                                            ui.label(egui::RichText::new(parts[1]).color(egui::Color32::DARK_GRAY));
                                                                        });
                                                                    } else {
                                                                        ui.label(egui::RichText::new(from).strong().color(header_color));
                                                                    }
                                                                } else {
                                                                    ui.label(egui::RichText::new(from).strong().color(header_color));
                                                                }
                                                            }
                                                            ui.label(egui::RichText::new(&item.content).color(content_color));
                                                        });
                                                    });
                                            });
                                        },
                                    );
                                    },
                                );
                            }

                            let is_waiting = *self.waiting_for_response.lock().unwrap();
                            if is_waiting {
                                ui.add_space(4.0);
                                ui.horizontal(|ui| {
                                    ui.add_space(70.0 + left_margin);
                                    ui.spinner();
                                });
                            }
                        });
                });
                
                // Add 4px spacing between scroll area and input panel
                ui.add_space(14.0);

                // Input panel at the bottom
                ui.add_space(-input_upward_spacing); // Add spacing from bottom (moves input upward)


                // Center the input row and make it 80% of chat width
                ui.with_layout(Layout::top_down(Align::Center), |ui| {
                    ui.set_max_width(ui.available_width() * 0.8);
                    let rounding = 8.0; // Half of previous 16.0
                    ui.horizontal(|ui| {
                        let control_height = 26.0;

                        // Text input (26px tall, rounded, vertically centered)
                        let available_for_input = ui.available_width() - 80.0; // Space for button + spacing
                        let input_frame = Frame::none()
                            .fill(ui.visuals().widgets.inactive.bg_fill)
                            .rounding(rounding)
                            .inner_margin(egui::Margin::symmetric(10.0, 3.0)); // Small vertical margin for centering
                        let response = input_frame
                            .show(ui, |ui| {
                                ui.set_height(control_height);
                                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                                    ui.add_sized(
                                        Vec2::new(available_for_input, control_height),
                                        egui::TextEdit::singleline(&mut self.input_text)
                                            .hint_text("Type a message")
                                            .frame(false),
                                    )
                                })
                                .inner
                            })
                            .inner;

                        ui.add_space(2.0); // Reduced spacing between input and button

                        // Send button (26px tall, rounded, vertically centered)
                        let button_frame = Frame::none()
                            .fill(ui.visuals().widgets.active.bg_fill)
                            .rounding(rounding)
                            .inner_margin(egui::Margin::symmetric(12.0, 3.0)); // Small vertical margin for centering
                        let send_button_response = button_frame
                            .show(ui, |ui| {
                                ui.set_height(control_height);
                                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                                    ui.add_sized(
                                        Vec2::new(40.0, control_height),
                                        egui::Button::new("Send").frame(false),
                                    )
                                })
                                .inner
                            })
                            .inner;
                        let send_button_clicked = send_button_response.clicked();
                        
                        // Handle Enter key or send button
                        let send_clicked = send_button_clicked
                            || (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));

                        if send_clicked && !self.input_text.trim().is_empty() {
                            let message_text = self.input_text.trim().to_string();
                            self.input_text.clear();

                            // Add user's message
                            let user_message = ChatMessage {
                                content: message_text.clone(),
                                from: Some("Human".to_string()),
                            };
                            self.messages.push(user_message);
                            self.message_timestamps.push(Self::current_timestamp_string());

                            // Use message handler if available, otherwise use default behavior
                            if let Some(handler) = &self.message_handler {
                                handler(message_text);
                            } else {
                                // Fallback: respond with "Please select a model"
                                let tx = self.inbox.sender();
                                let bot_message = ChatMessage {
                                    content: "Please select a model".to_string(),
                                    from: Some("System".to_string()),
                                };
                                tx.send(bot_message).ok();
                            }
                        }
                    });
                    
                    // Plus button under the input, aligned to left
                    ui.add_space(4.0); // Small spacing between input row and plus button
                    ui.with_layout(Layout::left_to_right(Align::Min), |ui| {
                        let plus_button_height = 26.0;
                        let plus_button_frame = Frame::none()
                            .fill(ui.visuals().widgets.inactive.bg_fill)
                            .rounding(rounding)
                            .inner_margin(egui::Margin::symmetric(12.0, 3.0));
                        let plus_button_response = plus_button_frame
                            .show(ui, |ui| {
                                ui.set_height(plus_button_height);
                                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                                    ui.add_sized(
                                        Vec2::new(10.0, plus_button_height),
                                        egui::Button::new("+").frame(false),
                                    )
                                })
                                .inner
                            })
                            .inner;
                        
                        if plus_button_response.clicked() {
                            // Open file dialog
                            if let Some(path) = rfd::FileDialog::new().pick_file() {
                                self.picked_file_path = Some(path.display().to_string());
                                println!("Selected file: {}", self.picked_file_path.as_ref().unwrap());
                            }
                        }
                        
                        // Display selected file path if any
                        if let Some(ref file_path) = self.picked_file_path {
                            ui.add_space(4.0);
                            ui.label(egui::RichText::new(format!("File: {}", file_path)).small().weak());
                        }
                    });
                });
        });
    }
}

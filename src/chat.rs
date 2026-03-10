use std::sync::Arc;

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
    pub fn new() -> Self {
        let history_loader = Arc::new(HistoryLoader::new());
        let inbox = UiInbox::new();
        
        // Load initial message from history
        let initial_messages = vec![
            ChatMessage {
                content: "Please select a model".to_string(),
                from: Some("System".to_string()),
            }
        ];

        ChatExample {
            messages: initial_messages,
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

    pub fn ui(&mut self, ui: &mut Ui) {
        // Read incoming messages from inbox
        self.inbox.read(ui).for_each(|message| {
            // Only add non-empty messages to prevent spacing issues
            if !message.content.trim().is_empty() {
                self.messages.push(message);
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

            Frame::none()
                .inner_margin(egui::Margin { left: 0.0, right: 0.0, top: top_padding, bottom: 0.0 })
                .show(ui, |ui| {
                    ScrollArea::vertical()
                                .animated(false)
                                .auto_shrink([false, false])
                                .stick_to_bottom(true)
                                .max_height(available_height - input_panel_height - top_padding - 20.0)
                                .show(ui, |ui| {
                                            ui.set_width(ui.available_width());

                        // Account for margins on each side - increase left margin if messages are too close to left
                        let left_margin = 10.0;  // Adjust this if messages need more space from left
                        let right_margin = 10.0;
                        // Maximum message width is 100% of parent width (minus margins)
                        let max_msg_width = ui.available_width() - left_margin - right_margin;

                        // Render messages with full control over spacing
                        // Direct rendering without extra layout wrappers to minimize spacing
                        for item in &self.messages {
                                let is_message_from_myself = item.from.as_deref() == Some("Human");

                                // Both Human and Bot messages align to the left
                                let layout = Layout::top_down(Align::Min);

                                ui.with_layout(layout, |ui| {
                                    // Allow messages to use full width (minus margins)
                                    ui.set_max_width(max_msg_width);

                                    let msg_color = if is_message_from_myself {
                                        ui.style().visuals.widgets.inactive.bg_fill
                                    } else {
                                        // All messages use black background
                                        ui.style().visuals.extreme_bg_color
                                    };
                                    
                                    // Determine border color for System and Agent Manager messages
                                    let border_color = match item.from.as_deref() {
                                        Some("System") => egui::Color32::from_rgb(204, 85, 0), // Dark orange
                                        Some("Agent Manager") => egui::Color32::from_rgb(0, 100, 0), // Dark green
                                        _ => egui::Color32::TRANSPARENT, // No border for other messages
                                    };
                                    
                                    let border_width = if border_color != egui::Color32::TRANSPARENT { 1.0 } else { 0.0 };

                                    let rounding = 4.0;
                                    let margin = 8.0;
                                    // 4px spacing between messages
                                    let outer_margin = egui::Margin {
                                        left: left_margin,   // 10px from left border
                                        right: right_margin, // 10px from right border
                                        top: 0.0,           // No extra space at top
                                        bottom: 4.0,        // 4px spacing after each message
                                    };
                                    
                                    // Calculate available width for content (accounting for margins)
                                    let content_max_width = max_msg_width - margin * 2.0;

                                    // Add 4px spacing between Agent Manager and the first row of content
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
                                            // All messages can use full width, text will wrap naturally
                                            ui.set_max_width(content_max_width);
                                            ui.with_layout(Layout::top_down(Align::Min), |ui| {
                                                // All messages use the same text colors (white header, gray content)
                                                let header_color = egui::Color32::WHITE;
                                                let content_color = egui::Color32::from_rgba_unmultiplied(150, 150, 150, 255);
                                                
                                                if let Some(from) = &item.from {
                                                    // For Ollama messages, show "Ollama" in white and model name in gray
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
                                                // Label automatically wraps when max_width is set
                                                ui.label(egui::RichText::new(&item.content).color(content_color));
                                            });
                                        });
                                });
                            }
                            
                            // Show spinner if waiting for response
                            let is_waiting = *self.waiting_for_response.lock().unwrap();
                            if is_waiting {
                                ui.add_space(4.0);
                                ui.horizontal(|ui| {
                                    ui.add_space(left_margin);
                                    ui.spinner();
                                });
                            }
                                    }); // Close ScrollArea
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

use serde::{Deserialize, Serialize};

use crate::chat::{AssistantGeneration, ChatMessage, MessageCorrelation};
use crate::incoming::MessageSource;

use super::Store;
use super::StoreError;

#[derive(Serialize, Deserialize)]
pub struct ConversationFile {
    pub schema_version: u32,
    pub conversation_id: String,
    pub messages: Vec<ExportedMessage>,
}

#[derive(Serialize, Deserialize)]
pub struct ExportedMessage {
    pub content: String,
    pub from: Option<String>,
    pub correlation: Option<MessageCorrelation>,
    pub source: String,
    pub api_auto_respond: bool,
    pub assistant_generation: Option<AssistantGeneration>,
    pub display_timestamp: String,
}

impl Store {
    pub fn export_conversation_json(&self, conversation_id: &str) -> Result<String, StoreError> {
        let (messages, timestamps) = self.load_messages(conversation_id)?;
        let mut out = Vec::with_capacity(messages.len());
        for (msg, ts) in messages.iter().zip(timestamps.iter()) {
            out.push(ExportedMessage {
                content: msg.content.clone(),
                from: msg.from.clone(),
                correlation: msg.correlation.clone(),
                source: msg.source.as_db().to_string(),
                api_auto_respond: msg.api_auto_respond,
                assistant_generation: msg.assistant_generation.clone(),
                display_timestamp: ts.clone(),
            });
        }
        let doc = ConversationFile {
            schema_version: 1,
            conversation_id: conversation_id.to_string(),
            messages: out,
        };
        Ok(serde_json::to_string_pretty(&doc)?)
    }

    /// Returns the new conversation id.
    pub fn import_conversation_json(&self, json: &str) -> Result<String, StoreError> {
        let file: ConversationFile = serde_json::from_str(json)?;
        let new_id = self.create_conversation()?;
        for m in file.messages {
            let source = MessageSource::from_db(&m.source).unwrap_or(MessageSource::System);
            let cm = ChatMessage {
                content: m.content,
                from: m.from,
                correlation: m.correlation,
                source,
                api_auto_respond: m.api_auto_respond,
                assistant_generation: m.assistant_generation,
            };
            self.append_message(&new_id, &cm, &m.display_timestamp)?;
        }
        Ok(new_id)
    }
}

//! SQLite persistence for conversations, messages, and assistant generation metadata.

mod export;
mod schema;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{params, Connection};
use rusqlite::OptionalExtension;

use crate::audit;
use crate::chat::{AssistantGeneration, ChatMessage, MessageCorrelation};
use crate::incoming::MessageSource;

pub use schema::USER_VERSION;

#[derive(Debug)]
pub enum StoreError {
    Sqlite(rusqlite::Error),
    Json(serde_json::Error),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Sqlite(e) => write!(f, "{e}"),
            StoreError::Json(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<rusqlite::Error> for StoreError {
    fn from(e: rusqlite::Error) -> Self {
        StoreError::Sqlite(e)
    }
}

impl From<serde_json::Error> for StoreError {
    fn from(e: serde_json::Error) -> Self {
        StoreError::Json(e)
    }
}

/// Per-conversation UI / generation defaults (persisted).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationSettings {
    pub selected_model: String,
    pub chat_token_limit: i32,
    pub chat_token_limit_enabled: bool,
    pub ollama_token_limit: i32,
    pub ollama_token_limit_enabled: bool,
}

impl Default for ConversationSettings {
    fn default() -> Self {
        Self {
            selected_model: String::new(),
            chat_token_limit: 70,
            chat_token_limit_enabled: false,
            ollama_token_limit: 70,
            ollama_token_limit_enabled: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConversationSummary {
    pub id: String,
    pub updated_at: String,
}

pub struct Store {
    conn: Mutex<Connection>,
    path: PathBuf,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref().to_path_buf();
        let conn = Connection::open(&path)?;
        let ver: i32 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap_or(0);
        if ver < USER_VERSION {
            schema::create_tables(&conn)?;
            // PRAGMA does not accept `?` placeholders in rusqlite/SQLite.
            conn.execute_batch(&format!("PRAGMA user_version = {USER_VERSION};"))?;
        }
        Ok(Self {
            conn: Mutex::new(conn),
            path,
        })
    }

    /// Load the latest conversation or create one. Empty message list means the UI should show the default welcome line.
    pub fn bootstrap_or_load(
        &self,
    ) -> Result<(String, Vec<ChatMessage>, Vec<String>, ConversationSettings), StoreError> {
        if let Some(id) = self.most_recent_conversation_id()? {
            let (msgs, ts) = self.load_messages(&id)?;
            let settings = self.load_conversation_settings(&id)?;
            Ok((id, msgs, ts, settings))
        } else {
            let id = self.create_conversation()?;
            Ok((id, Vec::new(), Vec::new(), ConversationSettings::default()))
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn create_conversation(&self) -> Result<String, StoreError> {
        let id = audit::new_id();
        let now = audit::now_rfc3339();
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO conversations (id, created_at, updated_at)
               VALUES (?1, ?2, ?3)"#,
            params![&id, &now, &now],
        )?;
        Ok(id)
    }

    /// Removes the conversation row; messages and generations cascade.
    pub fn delete_conversation(&self, id: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM conversations WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn save_conversation_settings(
        &self,
        id: &str,
        s: &ConversationSettings,
    ) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"UPDATE conversations SET
                selected_model = ?2,
                chat_token_limit = ?3,
                chat_token_limit_enabled = ?4,
                ollama_token_limit = ?5,
                ollama_token_limit_enabled = ?6,
                updated_at = ?7
               WHERE id = ?1"#,
            params![
                id,
                &s.selected_model,
                s.chat_token_limit,
                s.chat_token_limit_enabled as i32,
                s.ollama_token_limit,
                s.ollama_token_limit_enabled as i32,
                audit::now_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn load_conversation_settings(
        &self,
        id: &str,
    ) -> Result<ConversationSettings, StoreError> {
        let conn = self.conn.lock().unwrap();
        let s = conn.query_row(
            r#"SELECT selected_model, chat_token_limit, chat_token_limit_enabled,
                      ollama_token_limit, ollama_token_limit_enabled
               FROM conversations WHERE id = ?1"#,
            params![id],
            |r| {
                Ok(ConversationSettings {
                    selected_model: r.get(0)?,
                    chat_token_limit: r.get(1)?,
                    chat_token_limit_enabled: r.get::<_, i32>(2)? != 0,
                    ollama_token_limit: r.get(3)?,
                    ollama_token_limit_enabled: r.get::<_, i32>(4)? != 0,
                })
            },
        )?;
        Ok(s)
    }

    pub fn most_recent_conversation_id(&self) -> Result<Option<String>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let id: Option<String> = conn
            .query_row(
                "SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .optional()?;
        Ok(id)
    }

    pub fn list_conversations(&self, limit: usize) -> Result<Vec<ConversationSummary>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |r| {
            Ok(ConversationSummary {
                id: r.get(0)?,
                updated_at: r.get(1)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Append one message row; inserts into `generations` when `assistant_generation` is set.
    pub fn append_message(&self, conversation_id: &str, msg: &ChatMessage, display_ts: &str) -> Result<(), StoreError> {
        let correlation_json = msg
            .correlation
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"INSERT INTO messages (
                conversation_id, content, from_label, source, api_auto_respond,
                display_timestamp, correlation_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"#,
            params![
                conversation_id,
                &msg.content,
                msg.from.as_deref(),
                msg.source.as_db(),
                msg.api_auto_respond as i32,
                display_ts,
                correlation_json,
            ],
        )?;
        let mid = conn.last_insert_rowid();

        if let Some(ref g) = msg.assistant_generation {
            conn.execute(
                "INSERT INTO generations (message_id, model, num_predict) VALUES (?1, ?2, ?3)",
                params![mid, &g.model, g.num_predict],
            )?;
        }

        let now = audit::now_rfc3339();
        conn.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
            params![&now, conversation_id],
        )?;
        Ok(())
    }

    pub fn delete_messages_for_conversation(&self, conversation_id: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?1",
            params![conversation_id],
        )?;
        touch_conversation_conn(&conn, conversation_id)?;
        Ok(())
    }

    pub fn load_messages(
        &self,
        conversation_id: &str,
    ) -> Result<(Vec<ChatMessage>, Vec<String>), StoreError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"SELECT m.id, m.content, m.from_label, m.source, m.api_auto_respond,
                      m.display_timestamp, m.correlation_json,
                      g.model, g.num_predict
               FROM messages m
               LEFT JOIN generations g ON g.message_id = m.id
               WHERE m.conversation_id = ?1
               ORDER BY m.id ASC"#,
        )?;

        let rows = stmt.query_map(params![conversation_id], |r| {
            let correlation: Option<MessageCorrelation> = match r.get::<_, Option<String>>(6)? {
                Some(s) => Some(serde_json::from_str(&s).map_err(|e| {
                    rusqlite::Error::ToSqlConversionFailure(Box::new(e))
                })?),
                None => None,
            };
            let source = MessageSource::from_db(&r.get::<_, String>(3)?).unwrap_or(MessageSource::System);
            let api_ar = r.get::<_, i32>(4)? != 0;
            let display_ts: String = r.get(5)?;

            let assistant_generation =
                if let Some(model) = r.get::<_, Option<String>>(7)? {
                    Some(AssistantGeneration {
                        model,
                        num_predict: r.get(8)?,
                    })
                } else {
                    None
                };

            Ok((
                ChatMessage {
                    content: r.get(1)?,
                    from: r.get(2)?,
                    correlation,
                    source,
                    api_auto_respond: api_ar,
                    assistant_generation,
                },
                display_ts,
            ))
        })?;

        let mut messages = Vec::new();
        let mut timestamps = Vec::new();
        for row in rows {
            let (m, ts) = row?;
            messages.push(m);
            timestamps.push(ts);
        }
        Ok((messages, timestamps))
    }
}

fn touch_conversation_conn(conn: &Connection, id: &str) -> Result<(), StoreError> {
    let now = audit::now_rfc3339();
    conn.execute(
        "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
        params![&now, id],
    )?;
    Ok(())
}

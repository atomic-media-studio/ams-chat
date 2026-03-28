pub const USER_VERSION: i32 = 1;

pub fn create_tables(conn: &rusqlite::Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        r#"
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            selected_model TEXT NOT NULL DEFAULT '',
            chat_token_limit INTEGER NOT NULL DEFAULT 70,
            chat_token_limit_enabled INTEGER NOT NULL DEFAULT 0,
            ollama_token_limit INTEGER NOT NULL DEFAULT 70,
            ollama_token_limit_enabled INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            from_label TEXT,
            source TEXT NOT NULL,
            api_auto_respond INTEGER NOT NULL DEFAULT 0,
            display_timestamp TEXT NOT NULL,
            correlation_json TEXT
        );

        CREATE TABLE IF NOT EXISTS generations (
            message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
            model TEXT NOT NULL,
            num_predict INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
        "#,
    )?;
    Ok(())
}

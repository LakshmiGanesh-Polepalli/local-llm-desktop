CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,           -- Using UUID strings for Tauri/React compatibility
    title TEXT DEFAULT 'New Chat',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT                -- Track which model was used for this session
);

-- 2. Messages: Individual turn in the conversation
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT CHECK(role IN ('user', 'assistant')) NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexing conversation_id is CRITICAL for fetching a full thread quickly
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);

-- Trigger to automatically update the 'updated_at' column in conversations
CREATE TRIGGER IF NOT EXISTS update_conversations_timestamp 
AFTER INSERT ON messages
BEGIN
    UPDATE conversations 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.conversation_id;
END;
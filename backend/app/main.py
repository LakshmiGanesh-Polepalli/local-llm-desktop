import os
import re
import sys
import os
import json
import uuid
import httpx
import faiss
import psutil
import sqlite3
import threading
import requests
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import UploadFile, File
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware

# Force Python to look in the directory where main.py is located
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sandbox import execute_local_code
from services.document_processor import DocumentProcessor
from dotenv import load_dotenv
load_dotenv()

# 1. Add the global cancellation registry at the top of your file
cancel_registry = {}


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Create a permanent data folder in the Windows user directory
USER_HOME = os.path.expanduser("~")
APP_DIR = os.path.join(USER_HOME, ".sanctum-data")
os.makedirs(APP_DIR, exist_ok=True)

# Route the SQLite DB there
DB_FILE = os.path.join(APP_DIR, "sanctum.db")
REQUIRED_MODELS = ["llama3.2:1b"] 

def ensure_models_exist():
    """Silently checks and downloads missing models in the background."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return
            
        existing_models = [m["name"] for m in response.json().get("models", [])]
        
        for target_model in REQUIRED_MODELS:
            if target_model not in existing_models:
                print(f"[*] Missing target model '{target_model}'. Initiating background download...")
                requests.post(
                    "http://localhost:11434/api/pull",
                    json={"name": target_model, "stream": False},
                    timeout=3600
                )
                print(f"[+] Successfully installed {target_model}.")
                
    except requests.exceptions.RequestException:
        print("[!] Warning: Could not connect to Ollama. Ensure the Ollama app is running.")

class RenamePayload(BaseModel):
    title: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize FAISS index
    dim = 768
    index = faiss.IndexFlatL2(dim)

    # 2. Attach to app state
    app.state.vector_index = index
    app.state.doc_store = {} # To map index IDs back to conversation/message

    #3. BULLETPROOF DB INIT: Embedded schema guarantees tables exist on startup
   
    schema = """
    CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT DEFAULT 'New Chat',
    category TEXT DEFAULT 'Uncategorized', -- NEW: The metadata tag for Hybrid Search
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT
    );

    CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT CHECK(role IN ('user', 'assistant')) NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    );
    
    CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
    CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);
    """
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        # --- NEW: Safe Migration for existing databases ---
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN category TEXT DEFAULT 'Uncategorized'")
            print("Database migration successful: Added 'category' column.")
        except sqlite3.OperationalError:
            # Column already exists, safe to ignore
            pass

    # 4. RUN BACKFILL
    await backfill_index(app) # <--- ADD THIS CALL

    # 4. AUTO-PULL MODELS
    threading.Thread(target=ensure_models_exist, daemon=True).start()
        
    yield

app = FastAPI(title="LLM Desktop Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

async def generate_embedding(text: str) -> list[float]:
    """
    Calls the local Ollama engine to generate a vector embedding for the given text.
    Using nomic-embed-text yields a 768-dimensional vector.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": text
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return []
        
async def add_to_index(text: str, doc_id: str):
    """Embeds text and adds it to the FAISS index and doc_store."""
    vector = await generate_embedding(text)
    if not vector:
        return
    
    # Convert to float32 numpy array as required by FAISS
    vector_np = np.array([vector]).astype('float32')
    
    # Add to index
    app.state.vector_index.add(vector_np)
    
    # Store ID in doc_store (the index ID is simply the current count - 1)
    # This maps the FAISS numeric index to your DB message ID
    idx = app.state.vector_index.ntotal - 1
    app.state.doc_store[idx] = doc_id

async def backfill_index(app):
    """Runs on startup to index all existing messages."""
    print("Backfilling FAISS index...")
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT id, content FROM messages")
        rows = cur.fetchall()
        
        count = 0
        for row in rows:
            # Re-use your add_to_index logic
            # This generates the embedding and updates the state
            await add_to_index(row['content'], row['id'])
            count += 1
            
    print(f"Backfill complete. Indexed {count} messages.")

async def classify_conversation(conversation_id: str):
    """Silently categorizes a conversation in the background after a chat ends."""
    chat_text = ""
    
    # 1. Fetch the last 4 messages for context
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 4", 
            (conversation_id,)
        )
        rows = cur.fetchall()
        # Reverse so it reads chronologically
        for row in reversed(rows):
            chat_text += f"{row['role'].upper()}: {row['content']}\n"

    if not chat_text:
        return

    # 2. The Strict Classification Prompt
    prompt = f"""
    Analyze the following conversation and classify it into EXACTLY ONE of these categories:
    [Coding, Machine Learning, Hardware, Philosophy, General]

    Conversation:
    {chat_text}

    Reply ONLY with the exact category name. Do not add any other words or punctuation.
    """

    # 3. Call a stable LLM (Non-streaming)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3", # Highly recommended to use a standard model for background logic
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                
                # 4. Clean up the output (in case it hallucinates extra text)
                valid_categories = ["Coding", "Machine Learning", "Hardware", "Philosophy", "General"]
                final_category = "General" # Fallback
                
                for cat in valid_categories:
                    if cat.lower() in result.lower():
                        final_category = cat
                        break
                        
                # 5. Lock the tag into SQLite
                with sqlite3.connect(DB_FILE) as conn:
                    conn.execute("UPDATE conversations SET category = ? WHERE id = ?", (final_category, conversation_id))
                
                print(f"DEBUG: Background Task - Tagged chat [{conversation_id}] as [{final_category}]")
    except Exception as e:
        print(f"DEBUG: Background classification failed: {e}")

def get_hardware_status():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    return f"[System Monitor] CPU: {cpu}% | RAM: {ram}%"

# ==========================================
# API ROUTES
# ==========================================

@app.post("/api/chat/stream")
async def chat_stream(request: dict):
    prompt = request.get("prompt")
    
    # 🛡️ THE BOUNCER: Cap input to ~3500 tokens to save the GPU
    if len(prompt) > 15000:
        print(f"WARNING: Huge payload detected ({len(prompt)} chars). Truncating.")
        prompt = prompt[:15000] + "\n\n... [SYSTEM WARNING: INPUT TRUNCATED DUE TO HARDWARE MEMORY LIMITS]"

    model = request.get("model", "llama3")
    conversation_id = request.get("conversation_id", str(uuid.uuid4()))
    # Reset the cancel flag for this new stream
    cancel_registry[conversation_id] = False
    is_cloud = request.get("is_cloud", False) # NEW FLAG

    # --- 1. HYBRID RETRIEVAL ROUTER (Advanced RAG) ---
    relevant_context = ""
    detected_category = None
    
    # Fast keyword intent router to preserve low latency
    lower_prompt = prompt.lower()
    if "coding" in lower_prompt or "cpp" in lower_prompt or "c++" in lower_prompt or "dsa" in lower_prompt:
        detected_category = "Coding"
    elif "machine learning" in lower_prompt or "ml" in lower_prompt or "rag" in lower_prompt or "embedding" in lower_prompt:
        detected_category = "Machine Learning"
    elif "hardware" in lower_prompt or "gpu" in lower_prompt or "cuda" in lower_prompt:
        detected_category = "Hardware"
    elif "philosophy" in lower_prompt or "existential" in lower_prompt or "literature" in lower_prompt:
        detected_category = "Philosophy"

    snippets = []

    # Path A: Category Aggregation (If user asks about a specific domain history)
    if detected_category and any(kw in lower_prompt for kw in ["what", "list", "history", "past", "questions", "talked about"]):
        print(f"DEBUG: Hybrid Router - Category query detected for [{detected_category}]")
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            # Fetch user messages from conversations belonging to this specific category
            cur = conn.execute("""
                SELECT m.content FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.category = ? AND m.role = 'user'
                ORDER BY m.created_at DESC LIMIT 10
            """, (detected_category,))
            rows = cur.fetchall()
            for row in rows:
                if row['content'].strip().lower() != prompt.strip().lower():
                    snippets.append(f"[{detected_category} History]: {row['content']}")

    # Path B: Standard Semantic Vector Search (Fallback or Supplement)
    query_vector = await generate_embedding(prompt)
    if query_vector:
        query_np = np.array([query_vector]).astype('float32')
        # FIX: Bumped from 3 to 7 to bypass chat history pollution
        distances, indices = app.state.vector_index.search(query_np, 7) 
        
        for idx in indices[0]:
            if idx != -1 and idx in app.state.doc_store:
                msg_id = app.state.doc_store[idx]
                with sqlite3.connect(DB_FILE) as conn:
                    cur = conn.execute("SELECT content FROM messages WHERE id = ?", (msg_id,))
                    row = cur.fetchone()
                    if row and row[0] not in snippets and row[0].strip().lower() != prompt.strip().lower():
                        snippets.append(f"[Semantic Match]: {row[0]}")

    # Combine everything into the final injected context
    if snippets:
        relevant_context = "Relevant past context and history:\n" + "\n---\n".join(snippets) + "\n\n"
        print(f"DEBUG: Total Context Injected: {len(snippets)} snippets.")
        # FIX: Print exactly what is being fed to the LLM so we aren't blind
        print(f"DEBUG: INJECTED TEXT:\n{relevant_context}")

    # --- 2. SAVE USER MESSAGE (Once, before streaming) ---
    user_msg_id = str(uuid.uuid4())
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("INSERT OR IGNORE INTO conversations (id, title, model_used) VALUES (?, ?, ?)", 
                     (conversation_id, (prompt[:35] + "...") if len(prompt) > 35 else prompt, model))
        conn.execute("INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, 'user', ?)", 
                     (user_msg_id, conversation_id, prompt))
    
    await add_to_index(prompt, user_msg_id)
    

    async def pre_process_for_code(prompt: str):
        try:
            observations = []

            # 1. Hardware Monitor
            if any(keyword in prompt.lower() for keyword in ["status", "stats", "usage", "cpu", "ram"]):
                observations.append(get_hardware_status())

            # 2. Code Execution
            code_pattern = r"```(\w+)\s*\n([\s\S]*?)```"
            match = re.search(code_pattern, prompt)
            if match:
                lang = match.group(1).lower()
                code = match.group(2)
                if lang in ["python", "py", "cpp", "c++"]:
                    print(f"DEBUG: Running {lang} code...")
                    result = execute_local_code(lang, code)
                    observations.append(f"Execution Output: {result.get('stdout', '')} {result.get('stderr', '')}")
                    
            # Join everything safely
            return "\n".join(observations) if observations else ""
        
        except Exception as e:
            # If anything breaks, return the error so Sanctum can explain what went wrong
            print(f"CRITICAL ERROR in pre_process: {e}")
            return f"\n[Tool Execution Error]: {str(e)}"

    # --- 2.5 TOOL INTERCEPTOR ---
    execution_observation = await pre_process_for_code(prompt)

    # --- 3. STREAMING ---
    async def stream_generator():
        
        full_assistant_response = ""
        image_data = request.get("images", [])
        
        system_content = (
            "You are Sanctum, an intelligent local AI workstation assistant. "
            "Be concise and direct. "
            "If relevant <rag_memory_context> or <system_tools_output> is provided, you must prioritize it to answer. "
            "If the context is empty or irrelevant, answer the user naturally based on your own knowledge. "
            "Do not hallucinate safety violations."
        )
        
        augmented_user_prompt = (
            f"<system_tools_output>\n{execution_observation}\n</system_tools_output>\n\n"
            f"<rag_memory_context>\n{relevant_context}\n</rag_memory_context>\n\n"
            f"User Instruction: {prompt}"
        )
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": augmented_user_prompt}
        ]
        
        # Vision Path: Inject images if present
        if image_data:
            messages[1]["content"] = [{"type": "text", "text": augmented_user_prompt}]
            for img in image_data:
                 messages[1]["content"].append({"type": "image_url", "image_url": {"url": img}})

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                if is_cloud:
                    # ROUTE TO OPENROUTER
                    headers = {
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "Sanctum Workstation"
                    }
                    payload = {"model": model, "messages": messages, "stream": True}
                    
                    async with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            yield f"data: {json.dumps({'response': f'[Cloud Error]: HTTP {response.status_code} - {error_text.decode()}'})}\n\n"
                            yield f"data: {json.dumps({'done': True})}\n\n"
                            return

                        async for line in response.aiter_lines():
                            # CHECK FOR CANCELLATION
                            if cancel_registry.get(conversation_id, False):
                                yield f"data: {json.dumps({'response': ' [Stream Aborted]'})}\n\n"
                                break
                            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                                data = json.loads(line[6:])
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        full_assistant_response += delta
                                        yield f"data: {json.dumps({'response': delta})}\n\n"
                
                else:
                    # ROUTE TO LOCAL OLLAMA
                    payload = {"model": model, "messages": messages, "stream": True}
                    
                    # Ollama's default port is 11434
                    async with client.stream("POST", "http://localhost:11434/api/chat", json=payload) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            yield f"data: {json.dumps({'response': f'[Local Error]: HTTP {response.status_code} - {error_text.decode()}'})}\n\n"
                            yield f"data: {json.dumps({'done': True})}\n\n"
                            return

                        # Ollama returns JSON lines natively, not SSE
                        async for line in response.aiter_lines():
                            if cancel_registry.get(conversation_id, False):
                                yield f"data: {json.dumps({'response': ' [Stream Aborted]'})}\n\n"
                                break
                            if line:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    delta = data["message"]["content"]
                                    full_assistant_response += delta
                                    # Wrap it in the exact SSE format the frontend expects
                                    yield f"data: {json.dumps({'response': delta})}\n\n"
                                    
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"

        # SAVE ASSISTANT MESSAGE (After stream finishes)
        if full_assistant_response.strip():
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, 'assistant', ?)", 
                             (str(uuid.uuid4()), conversation_id, full_assistant_response))

    # Create the background task pointing to our new function
    task = BackgroundTask(classify_conversation, conversation_id=conversation_id)
    
    # Return the stream, and FastAPI will automatically run 'task' when the stream finishes
    return StreamingResponse(
        stream_generator(), 
        media_type="text/event-stream", 
        background=task
    )

@app.post("/api/chat/stop")
async def stop_generation(request: dict):
    conversation_id = request.get("conversation_id")
    if conversation_id:
        cancel_registry[conversation_id] = True
        return {"status": "cancelled"}
    return {"status": "ignored"}

# --- NEW: DATABASE FETCH ROUTES ---

@app.get("/api/test-embedding/{text}")
async def test_embedding(text: str):
    vector = await generate_embedding(text)
    if not vector:
        raise HTTPException(status_code=500, detail="Embedding failed")
    
    return {
        "text": text,
        "dimension": len(vector),
        "sample": vector[:5] # Return first 5 floats to verify it's a valid array
    }

@app.get("/api/search")
async def search_memory(query: str, top_k: int = 3):
    """Retrieve relevant conversation history based on the query."""
    # 1. Generate embedding for query
    query_vector = await generate_embedding(query)
    if not query_vector:
        return {"results": []}

    query_np = np.array([query_vector]).astype('float32')

    # 2. Search FAISS
    distances, indices = app.state.vector_index.search(query_np, top_k)
    
    # 3. Fetch actual text from SQLite
    results = []
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        for idx in indices[0]:
            if idx in app.state.doc_store:
                msg_id = app.state.doc_store[idx]
                cur = conn.execute("SELECT content FROM messages WHERE id = ?", (msg_id,))
                row = cur.fetchone()
                if row:
                    results.append(row['content'])
                    
    return {"results": results}

@app.get("/api/conversations")
def get_conversations():
    """Fetch all conversation threads for the sidebar"""
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # FIX: Explicitly hide the 'system_files' thread so it can't be deleted via UI
        cur.execute("SELECT id, title, updated_at FROM conversations WHERE id != 'system_files' ORDER BY updated_at DESC")
        return [dict(row) for row in cur.fetchall()]

@app.get("/api/conversations/{conversation_id}")
def get_messages(conversation_id: str):
    """Fetch all messages for a specific conversation"""
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT id, role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC", (conversation_id,))
        return [dict(row) for row in cur.fetchall()]

@app.get("/api/models")
async def get_models():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:11434/api/tags")
            data = response.json()
            excluded = ["nomic-embed-text", "mxbai-embed", "all-minilm"]
            data["models"] = [
                m for m in data.get("models", [])
                if not any(e in m["name"].lower() for e in excluded)
            ]
            return data
        except Exception:
            return {"models": []}
        
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Extracts text, chunks it, and injects it into FAISS AND SQLite."""
    try:
        contents = await file.read()
        ext = file.filename.split('.')[-1]
        
        # Extract and Chunk
        raw_text = DocumentProcessor.extract_text_from_bytes(contents, ext)
        chunks = DocumentProcessor.chunk_text(raw_text)
        
        with sqlite3.connect(DB_FILE) as conn:
            # 1. Create a hidden "system" conversation to satisfy the foreign key constraint
            conn.execute(
                "INSERT OR IGNORE INTO conversations (id, title, category) VALUES (?, ?, ?)", 
                ('system_files', 'Uploaded Documents', 'System')
            )
            
            for chunk in chunks:
                doc_uuid = f"file_{uuid.uuid4()}"
                chunk_text = f"[File Content from {file.filename}]: {chunk['text']}"
                
                # 2. SAVE TEXT TO SQLITE (This is what was missing!)
                conn.execute(
                    "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, 'system_files', 'user', ?)", 
                    (doc_uuid, chunk_text)
                )
                
                # 3. Add Vector to FAISS
                await add_to_index(chunk_text, doc_uuid)
                
        return {"status": "success", "chunks_indexed": len(chunks), "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "operational", "engine": "FastAPI"}

# --- UPDATE & DELETE ROUTES ---

@app.delete("/api/chat/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages"""
    with sqlite3.connect(DB_FILE) as conn:
        # Your schema has ON DELETE CASCADE for messages, but running both is safe
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
    return {"status": "success", "deleted_id": conversation_id}

@app.patch("/api/chat/{conversation_id}")
def rename_conversation(conversation_id: str, payload: RenamePayload):
    """Rename a specific conversation thread"""
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE conversations SET title = ? WHERE id = ?", (payload.title, conversation_id))
        conn.commit()
        
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
    return {"status": "success", "new_title": payload.title}


if __name__ == "__main__":
    import uvicorn
    import sys
    
    # This keeps the executable running as a server on port 8000
    print("[*] Starting Sanctum Backend Sidecar on port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)



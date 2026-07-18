# Hybrid Desktop LLM Workspace

A high-performance, locally-first hybrid desktop application for Large Language Models. Built to leverage local workstation hardware for zero-latency inference and RAG, with seamless cloud fallback capabilities.

## 🏗 Architecture

**Frontend:**
* Desktop Runtime: Tauri
* UI Framework: React + Tailwind CSS (Gemini-style single-window interface)

**Backend:**
* Server: FastAPI (Python)
* Communication: REST & Server-Sent Events (SSE) for streaming

**Inference & Memory:**
* Local Engine: Ollama (GPU accelerated)
* Cloud Fallback: OpenRouter (OpenAI-compatible)
* Embeddings: `nomic-embed-text` (CPU/RAM via num_gpu=0)
* Vector Store: FAISS (In-RAM operations, SQLite persistence on close)

## 🚀 Current Status

* [x] Stack Definition
* [x] FastAPI Server Scaffold
* [x] Local Ollama Connection
* [ ] Tauri + React UI Setup
* [ ] FAISS + SQLite RAG Pipeline
* [ ] OpenRouter Fallback Logic

## 🛠 Prerequisites

1. Install [Ollama](https://ollama.com/) natively on your workstation.
2. Pull the required models:
   ```bash
   ollama run llama3
   ollama pull nomic-embed-text

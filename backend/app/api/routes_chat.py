from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.ollama_client import stream_ollama_response

router = APIRouter()

class ChatRequest(BaseModel):
    prompt: str
    model: str | None = None

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    model_to_use = request.model if request.model else "llama3"
    
    return StreamingResponse(
        stream_ollama_response(prompt=request.prompt, model=model_to_use),
        media_type="text/event-stream"
    )
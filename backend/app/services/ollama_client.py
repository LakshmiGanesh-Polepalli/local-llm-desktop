import httpx
import json
from app.core.config import settings

async def stream_ollama_response(prompt: str, model: str = settings.DEFAULT_MODEL):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("POST", f"{settings.OLLAMA_BASE_URL}/generate", json=payload, timeout=None) as response:
                # If Ollama returns a 404 or 500, this triggers the exception block below
                response.raise_for_status() 
                
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("response", "")
                        yield f"data: {json.dumps({'text': token})}\n\n"
                        
                        if data.get("done"):
                            break
                            
        except httpx.HTTPStatusError as e:
            # Catch 404s (Missing Model) or other HTTP errors gracefully
            yield f"data: {json.dumps({'error': f'Ollama HTTP Error {e.response.status_code}. Is the model downloaded?'})}\n\n"
        except httpx.ConnectError:
            # Catch network bridge failures gracefully
            yield f"data: {json.dumps({'error': 'Failed to connect to Ollama. Is it running?'})}\n\n"
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Local LLM Backend"
    OLLAMA_BASE_URL: str = "http://localhost:11434/api"
    DEFAULT_MODEL: str = "llama3"

settings = Settings()
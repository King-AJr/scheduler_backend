from pydantic_settings import BaseSettings
from functools import lru_cache
import os

MODEL_NAME = os.getenv("MODEL_NAME")

class Settings(BaseSettings):
    PROJECT_NAME: str = "Scheduler Assistant"
    FIREBASE_CREDENTIALS: str
    FIREBASE_WEB_API_KEY: str
    GROQ_API_KEY: str
    MODEL_NAME: str = MODEL_NAME
    PINECONE_API_KEY: str
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
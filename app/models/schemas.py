from pydantic import BaseModel
from typing import Optional

class ChatMessage(BaseModel):
    content: str
    role: str = "user"

class ChatResponse(BaseModel):
    message: str 
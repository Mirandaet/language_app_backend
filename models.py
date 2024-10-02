from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatHistory(BaseModel):
    chat_history: List[ChatMessage]
    language: str
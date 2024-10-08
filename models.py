from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

class ChatMessage(BaseModel):
    role: str
    message: str

class ChatHistory(BaseModel):
    chat_history: List[ChatMessage]
    language: str
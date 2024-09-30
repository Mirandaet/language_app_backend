from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class ChatHistory(BaseModel):
    chat_history: List[dict]
    language: str
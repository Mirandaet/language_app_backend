from pydantic import BaseModel
from typing import List
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv


class ChatMessage(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    user_id: int
    conversation_id: int
    message: str
    language: str

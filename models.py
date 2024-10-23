from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Boolean, LargeBinary
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    conversations = relationship("Conversation", back_populates="user")

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    language = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    is_user_message = Column(Boolean, nullable=False)
    is_summary = Column(Boolean, default=False)
    summary_start_time = Column(DateTime(timezone=True))
    summary_end_time = Column(DateTime(timezone=True))
    conversation = relationship("Conversation", back_populates="messages")
    embedding = relationship("MessageEmbedding", uselist=False, back_populates="message")

class MessageEmbedding(Base):
    __tablename__ = 'message_embeddings'
    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey('messages.id'))
    embedding = Column(LargeBinary, nullable=False)
    message = relationship("Message", back_populates="embedding")

class FAISSIndex(Base):
    __tablename__ = 'faiss_indices'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    index_data = Column(LargeBinary, nullable=False)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user = relationship("User")

# Pydantic models (for API)
class ChatMessage(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    user_id: int
    conversation_id: int
    message: str
    language: str

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
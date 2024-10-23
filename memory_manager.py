from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from lms.ollama import llama
import logging
from sqlalchemy.orm import Session
from models import User, Conversation, Message
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class MemoryManager:
    _instance = None

    def __new__(cls, db: Session, window_size=10, chunk_size=10, max_summaries=20):
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db: Session, window_size=10, chunk_size=50, max_summaries=20):
        if self._initialized:
            return
        self.db = db
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.max_summaries = max_summaries
        self.overlap_size = self.chunk_size // 2

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.user_indices = {}

        logger.info(f"MemoryManager initialized with window_size={window_size}, chunk_size={chunk_size}, max_summaries={max_summaries}")
        self._initialized = True

    def set_db(self, db: Session):
        self.db = db

    def get_or_create_user_index(self, user_id):
        if user_id not in self.user_indices:
            self.user_indices[user_id] = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Created new FAISS index for user {user_id}")
        return self.user_indices[user_id]

    def add_message_to_store(self, content, conversation_id, user_id, is_user_message=True, is_summary=False, start_time=None, end_time=None):
        try:
            conversation = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conversation:
                # Create a new conversation if it doesn't exist
                user = self.db.query(User).filter(User.id == user_id).first()
                if not user:
                    user = User(id=user_id, username=f"user_{user_id}")  # Create a new user if not exists
                    self.db.add(user)
                
                conversation = Conversation(id=conversation_id, user_id=user_id, language="English")  # Default language to English
                self.db.add(conversation)
                self.db.commit()
                logger.info(f"Created new conversation with id {conversation_id} for user {user_id}")

            message = Message(
                conversation_id=conversation_id,
                content=content,
                is_user_message=is_user_message,
                is_summary=is_summary,
                summary_start_time=start_time,
                summary_end_time=end_time
            )
            self.db.add(message)
            self.db.commit()

            self.index_message_for_rag(message)

            logger.debug(f"Added message to database: user_id={user_id}, conversation_id={conversation_id}, message_id={message.id}")
            return message.id
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error occurred: {str(e)}")
            raise

    def retrieve_relevant_messages(self, query, user_id, k=5):
        query_embedding = self.embedding_model.encode([query])[0]
        user_index = self.get_or_create_user_index(user_id)
        distances, indices = user_index.search(np.array([query_embedding], dtype=np.float32), k)
        
        relevant_messages = self.db.query(Message).filter(
            Message.id.in_(indices[0]),
            Message.conversation.has(Conversation.user_id == user_id)
        ).all()
        
        logger.debug(f"Retrieved {len(relevant_messages)} relevant messages for user {user_id}")
        return relevant_messages

    def get_conversation_window(self, conversation_id, user_id):
        logger.debug(f"Retrieving conversation window for user {user_id}, conversation {conversation_id}")
        
        # Get the last summarized message for this conversation
        last_summary = self.db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.is_summary == True
        ).order_by(Message.id.desc()).first()

        if last_summary:
            messages = self.db.query(Message).filter(
                Message.conversation_id == conversation_id,
                Message.id > last_summary.id
            ).order_by(Message.id).all()
        else:
            messages = self.db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.id).all()

        # Include recent summaries
        summaries = self.db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.is_summary == True
        ).order_by(Message.id.desc()).limit(self.max_summaries).all()

        context_messages = summaries[::-1] + messages

        # Ensure we don't exceed window_size + chunk_size
        max_context_size = self.window_size + self.chunk_size
        if len(context_messages) > max_context_size:
            context_messages = context_messages[-max_context_size:]

        return context_messages

    def manage_conversation_window(self, conversation_id, user_id):
        logger.debug(f"Managing conversation window for user {user_id}, conversation {conversation_id}")
        
        unsummarized_messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.is_summary == False
        ).order_by(Message.id).all()

        if len(unsummarized_messages) >= self.window_size + self.chunk_size:
            messages_to_summarize = unsummarized_messages[:len(unsummarized_messages) - self.window_size]
            summary, start_time, end_time = self.summarize_chunk(messages_to_summarize, user_id)
            self.add_message_to_store(summary, conversation_id, user_id, is_user_message=False, is_summary=True, 
                                      start_time=start_time, end_time=end_time)
            
            self.manage_summaries(conversation_id, user_id)

    def manage_summaries(self, conversation_id, user_id):
        summaries = self.db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.is_summary == True
        ).order_by(Message.summary_start_time).all()

        if len(summaries) > self.max_summaries:
            oldest_summaries = summaries[:len(summaries) - self.max_summaries + 1]
            merged_summary, start_time, end_time = self.summarize_chunk(oldest_summaries, user_id)
            self.add_message_to_store(merged_summary, conversation_id, user_id, is_user_message=False, is_summary=True, 
                                      start_time=start_time, end_time=end_time)
            
            for summary in oldest_summaries:
                self.db.delete(summary)
            self.db.commit()

            logger.info(f"Merged {len(oldest_summaries)} old summaries for user {user_id}, conversation {conversation_id}")

    def summarize_chunk(self, messages, user_id):
        context = "\n".join([f"{msg.timestamp}: {'User' if msg.is_user_message else 'AI'}: {msg.content}" for msg in messages])
        start_time = messages[0].timestamp
        end_time = messages[-1].timestamp
        
        prompt = f"""
        Summarize the following chunk of conversation that occurred between {start_time} and {end_time}, 
        capturing the main points and any important details:

        {context}

        Summary:
        """
        summary = llama(system_prompt=f"You are summarizing a conversation for user {user_id}.", prompt=prompt)
        logger.debug(f"Summarized chunk of {len(messages)} messages for user {user_id}")
        return summary.strip(), start_time, end_time

    def index_message_for_rag(self, message):
        user_id = message.conversation.user_id
        user_index = self.get_or_create_user_index(user_id)
        
        if message.is_summary:
            text_to_embed = f"Summary from {message.summary_start_time} to {message.summary_end_time}: {message.content}"
        else:
            text_to_embed = f"{message.timestamp}: {'User' if message.is_user_message else 'AI'}: {message.content}"
        
        embedding = self.embedding_model.encode([text_to_embed])[0]
        user_index.add(np.array([embedding], dtype=np.float32))
        
        logger.debug(f"Indexed message for RAG: user_id={user_id}, message_id={message.id}")

    def search_conversation(self, query, user_id):
        return self.retrieve_relevant_messages(query, user_id)

    def retrieve_messages_by_timeframe(self, start_time, end_time, user_id):
        relevant_messages = self.db.query(Message).join(Conversation).filter(
            Conversation.user_id == user_id,
            ((Message.is_summary == False) & (Message.timestamp.between(start_time, end_time))) |
            ((Message.is_summary == True) & 
             ((Message.summary_start_time <= end_time) & (Message.summary_end_time >= start_time)) |
             (Message.timestamp.between(start_time, end_time)))
        ).all()
        
        logger.debug(f"Retrieved {len(relevant_messages)} messages by timeframe for user {user_id}")
        return relevant_messages
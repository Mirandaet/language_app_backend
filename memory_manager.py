from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from lms.ollama import llama
import math
import logging
import threading

# Set up logging
logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, window_size=10, chunk_size=50, max_summaries=20):
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.max_summaries = max_summaries
        self.overlap_size = self.chunk_size // 2  # Half of the summarized messages are reused

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.user_indices = {}
        self.user_id_to_message = {}
        self.user_current_id = {}
        self.last_summarized_id = {}  # Track the ID of the last summarized message for each user

        self.instance_id = id(self)
        self.thread_id = threading.get_ident()

        logger.info(f"MemoryManager initialized with window_size={window_size}, chunk_size={chunk_size}, max_summaries={max_summaries}")
        logger.info(f"Instance ID: {self.instance_id}, Thread ID: {self.thread_id}")
        logger.debug(f"Initial user_current_id state: {self.user_current_id}")
        logger.debug(f"Initial user_id_to_message state: {self.user_id_to_message}")

    def get_or_create_user_index(self, user_id):
        """Get or create a FAISS index for a specific user."""
        if user_id not in self.user_indices:
            self.user_indices[user_id] = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Created new FAISS index for user {user_id}")
        return self.user_indices[user_id]

    def add_message_to_store(self, content, conversation_id, user_id, is_user_message=True, is_summary=False, start_time=None, end_time=None):
        """Add a message or summary to the store and index it for RAG."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.debug(f"add_message_to_store called. Instance ID: {id(self)}, Thread ID: {threading.get_ident()}")
        logger.debug(f"Current user_current_id state: {self.user_current_id}")
        logger.debug(f"Current user_id_to_message state: {self.user_id_to_message}")
        
        if user_id not in self.user_current_id:
            self.user_current_id[user_id] = 0
            logger.debug(f"Initialized user_current_id for user {user_id} to -1")
        
        self.user_current_id[user_id] += 1
        logger.debug(f"Incremented user_current_id for user {user_id}")
        message_id = self.user_current_id[user_id]
        
        logger.debug(f"Assigned message_id: {message_id}")
        logger.debug(f"Incremented user_current_id for user {user_id} to: {self.user_current_id[user_id]}")
        
        logger.debug(f"Adding message to store: user_id={user_id}, conversation_id={conversation_id}, message_id={message_id}, is_user_message={is_user_message}, message={content}")
        
        message = {
            'id': message_id,
            'content': content,
            'timestamp': timestamp,
            'conversation_id': conversation_id,
            'user_id': user_id,
            'is_user_message': is_user_message,
            'is_summary': is_summary
        }
        if is_summary:
            message['summary_start_time'] = start_time
            message['summary_end_time'] = end_time
        
        if user_id not in self.user_id_to_message:
            self.user_id_to_message[user_id] = {}
            logger.debug(f"Created new message store for user {user_id}")
        
        self.user_id_to_message[user_id][message_id] = message
        logger.debug(f"Added message to user store for user {user_id}")
        
        self.index_message_for_rag(message)
        
        logger.debug(f"Final user_current_id state: {self.user_current_id}")
        logger.debug(f"Final user_id_to_message state for user {user_id}: {self.user_id_to_message[user_id]}")
        logger.debug(f"Total messages for user {user_id}: {len(self.user_id_to_message[user_id])}")
        
        return message_id
    def retrieve_relevant_messages(self, query, user_id, k=5):
        """Retrieve k most relevant messages for a given query and user."""
        query_embedding = self.embedding_model.encode([query])[0]
        
        user_index = self.get_or_create_user_index(user_id)
        
        distances, indices = user_index.search(np.array([query_embedding], dtype=np.float32), k)
        
        relevant_messages = [self.user_id_to_message[user_id][i] for i in indices[0] if i in self.user_id_to_message[user_id]]
        
        logger.debug(f"Retrieved {len(relevant_messages)} relevant messages for user {user_id}")
        return relevant_messages

    def get_conversation_window(self, conversation_id, user_id):
        logger.debug(f"Retrieving conversation window for user {user_id}, conversation {conversation_id}")
        user_messages = self.user_id_to_message.get(user_id, {})
        logger.debug(f"Retrieved {len(user_messages)} messages for user {user_id}")

        # Get the last summarized message ID for this user
        last_summarized_id = self.last_summarized_id.get(user_id, -1)
        logger.debug(f"Last summarized ID for user {user_id}: {last_summarized_id}")

        # Filter and sort messages for this conversation

        all_messages = sorted([msg for msg in user_messages.values() if msg['conversation_id'] == conversation_id],
                            key=lambda x: x['id'])
        logger.debug(f"Retrieved all messages: {all_messages}")


        # Get unsummarized messagess
        unsummarized_messages = [msg for msg in all_messages if msg['id'] > last_summarized_id and not msg['is_summary']]
        
        logger.debug(f"Retrieved {len(unsummarized_messages)} unsummarized messages for user {user_id}, conversation {conversation_id}")

        # Get recent summaries
        summaries = [msg for msg in all_messages if msg['is_summary']]
        recent_summaries = summaries[-self.max_summaries:] if summaries else []

        # Combine recent summaries and unsummarized messages
        context_messages = recent_summaries + unsummarized_messages

        # Ensure we don't exceed window_size + chunk_size
        max_context_size = self.window_size + self.chunk_size
        if len(context_messages) > max_context_size:
            context_messages = context_messages[-max_context_size:]

        return context_messages

    def manage_conversation_window(self, conversation_id, user_id):
        """Manages the sliding window and chunked summarization for conversation history."""
        logger.debug(f"Managing conversation window for user {user_id}, conversation {conversation_id}")
        user_messages = self.user_id_to_message.get(user_id, {})
        
        logger.debug(f"Retrieved {len(user_messages)} user_messages for user {user_id}")
        # Filter messages for this conversation
        conversation_messages = [msg for msg in user_messages.values() 
                                 if msg['conversation_id'] == conversation_id and not msg['is_summary']]
        
        logger.debug(f"Retrieved {len(conversation_messages)} conversation_messages for user {user_id}, conversation {conversation_id}")
        # Sort messages by their ID to ensure chronological order
        conversation_messages.sort(key=lambda x: x['id'])
        
        last_summarized_id = self.last_summarized_id.get(user_id, -1)
        if last_summarized_id == -1:
            logger.debug(f"No last summarized ID found for user {user_id}")
            unsummarized_messages = conversation_messages
            logger.debug(f"Retrieved {len(unsummarized_messages)} unsummarized messages")
        else:
            logger.debug(f"Last summarized ID found for user {user_id}: {last_summarized_id}")
            unsummarized_messages = [msg for msg in conversation_messages if msg['id'] > last_summarized_id]
            logger.debug(f"Retrieved {len(unsummarized_messages)} unsummarized messages")
        
        total_unsummarized = len(unsummarized_messages)

        if total_unsummarized >= self.window_size + self.chunk_size:
            # Summarize all unsummarized messages
            messages_to_summarize = unsummarized_messages[:total_unsummarized - self.window_size]
            
            summary, start_time, end_time = self.summarize_chunk(messages_to_summarize, user_id)
            self.add_message_to_store(summary, conversation_id, user_id, is_user_message=False, is_summary=True, 
                                      start_time=start_time, end_time=end_time)
            
            # Update the last summarized ID
            self.last_summarized_id[user_id] = messages_to_summarize[-1]['id']
            
            # Manage the number of summaries
            self.manage_summaries(conversation_id, user_id)

    def manage_summaries(self, conversation_id, user_id):
        """Manages the number of summaries, merging oldest ones if necessary."""
        summaries = [msg for msg in self.user_id_to_message[user_id].values() 
                     if msg['is_summary'] and msg['conversation_id'] == conversation_id]
        
        if len(summaries) > self.max_summaries:
            summaries.sort(key=lambda x: x['summary_start_time'])
            
            oldest_summaries = summaries[:len(summaries) - self.max_summaries + 1]
            merged_summary, start_time, end_time = self.summarize_chunk(oldest_summaries, user_id)
            self.add_message_to_store(merged_summary, conversation_id, user_id, is_user_message=False, is_summary=True, 
                                      start_time=start_time, end_time=end_time)
            
            for summary in oldest_summaries:
                self.user_id_to_message[user_id].pop(summary['id'], None)
            
            logger.info(f"Merged {len(oldest_summaries)} old summaries for user {user_id}, conversation {conversation_id}")

    def summarize_chunk(self, messages, user_id):
        """Summarize a chunk of messages with timestamp information."""
        context = "\n".join([f"{msg['timestamp']}: {'User' if msg['is_user_message'] else 'AI'}: {msg['content']}" for msg in messages])
        start_time = messages[0]['timestamp']
        end_time = messages[-1]['timestamp']
        
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
        """Index the message for RAG retrieval."""
        user_id = message['user_id']
        
        user_index = self.get_or_create_user_index(user_id)
        
        if message['is_summary']:
            text_to_embed = f"Summary from {message['summary_start_time']} to {message['summary_end_time']}: {message['content']}"
        else:
            text_to_embed = f"{message['timestamp']}: {'User' if message['is_user_message'] else 'AI'}: {message['content']}"
        
        embedding = self.embedding_model.encode([text_to_embed])[0]
        
        user_index.add(np.array([embedding], dtype=np.float32))
        
        logger.debug(f"Indexed message for RAG: user_id={user_id}, message_id={message['id']}")

    def search_conversation(self, query, user_id):
        """Searches through the conversation using RAG and retrieves relevant parts"""
        return self.retrieve_relevant_messages(query, user_id)

    def retrieve_messages_by_timeframe(self, start_time, end_time, user_id):
        """Retrieve messages and summaries within a specific timeframe for a specific user."""
        user_messages = self.user_id_to_message.get(user_id, {})
        
        relevant_messages = [msg for msg in user_messages.values() if 
                (not msg['is_summary'] and start_time <= msg['timestamp'] <= end_time) or
                (msg['is_summary'] and 
                 ((msg['summary_start_time'] <= end_time and msg['summary_end_time'] >= start_time) or
                  (start_time <= msg['timestamp'] <= end_time)))]
        
        logger.debug(f"Retrieved {len(relevant_messages)} messages by timeframe for user {user_id}")
        return relevant_messages


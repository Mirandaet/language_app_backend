# Import necessary libraries
from openai import OpenAI
import os
from queue import Queue
from datetime import datetime
import json
import torch
import librosa
import io
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer,  SetLogLevel
import wave
import logging
import argparse
from pydub import AudioSegment
from pydub.effects import normalize
import whisper
from lms.ollama import llama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from memory_manager import MemoryManager
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

# Load the Sentence-BERT model for generating embeddings
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
logger.info("Sentence-BERT model loaded")

# FAISS index (we'll use L2 distance, but you could use others like cosine similarity)
dimension = 384  # Embedding dimensionality for MiniLM
index = faiss.IndexFlatL2(dimension)
logger.info(f"FAISS index initialized with dimension {dimension}")

# Store the messages alongside their conversation IDs and user IDs
messages_store = []

memory_manager = MemoryManager(window_size=10)
logger.info(f"Memory manager initialized with window size 10. Instance ID: {id(memory_manager)}")

def load_system_prompt(language):
    """Load the system prompt from the languages_desc.json file based on the language."""
    try:
        with open("languages_desc.json", "r") as file:
            languages = json.load(file)
            if language in languages:
                # Return the system prompt for the language
                return languages[language][0]
            else:
                logger.error(
                    f"Language '{language}' not found in languages_desc.json")
                raise ValueError(
                    f"Language '{language}' not found in languages_desc.json")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise Exception(f"Error loading languages_desc.json: {str(e)}")


def transcribe_audio(content, language, input_format='webm'):
    logger.info(f"Starting audio transcription. Language: {language}, Format: {input_format}")

    # Load the Whisper model
    model = whisper.load_model("small")
    logger.debug("Whisper model loaded")

    # Convert the byte content to an audio segment using pydub
    audio_segment = AudioSegment.from_file(
        io.BytesIO(content), format=input_format)
    logger.debug("Audio content converted to AudioSegment")

    # Resample to 16000 Hz as required by Whisper
    audio_segment = audio_segment.set_frame_rate(16000)
    logger.debug("Audio resampled to 16000 Hz")

    # Convert to mono if stereo
    if audio_segment.channels == 2:
        audio_segment = audio_segment.set_channels(1)
        logger.debug("Audio converted to mono")

    # Convert the audio segment to a numpy array
    samples = np.array(audio_segment.get_array_of_samples()
                       ).flatten().astype(np.float32)
    logger.debug("Audio converted to numpy array")

    # Normalize
    samples /= np.max(np.abs(samples))
    logger.debug("Audio normalized")

    # Transcribe the audio
    logger.info("Starting Whisper transcription")
    result = model.transcribe(samples, language=language)
    logger.info(f"Transcription completed. Result: {result['text']}")

    return result


def generate_response(new_message, conversation_id, user_id, language="English"):
    logger.info(f"Generating response for user {user_id}, conversation {conversation_id}")
    logger.debug(f"New message: {new_message}")
    logger.debug(f"Language: {language}")
    logger.debug(f"Using MemoryManager instance ID: {id(memory_manager)}")

    system_prompt = load_system_prompt(language)
    logger.debug(f"System prompt: {system_prompt}")

    # Retrieve all relevant messages within the current window

    # First, add the new message to the conversation window

    window_messages = memory_manager.get_conversation_window(conversation_id, user_id)
    logger.debug(f"Retrieved {len(window_messages)} messages from the conversation window")
    
    user_message_id = memory_manager.add_message_to_store(new_message, conversation_id, user_id, is_user_message=True)
    logger.debug(f"Added new user message with ID: {user_message_id}")

    # Manage conversation window (store new message, summarize old ones if necessary)
    memory_manager.manage_conversation_window(conversation_id, user_id)
    logger.debug("Conversation window managed")

    # Prepare context with proper classification
    context = []
    for msg in window_messages:
        if msg['is_summary']:
            context.append(f"Summary ({msg['summary_start_time']} to {msg['summary_end_time']}): {msg['content']}")
        elif msg['is_user_message']:
            context.append(f"User: {msg['content']}")
        else:
            context.append(f"AI: {msg['content']}")
    
    context_str = "\n".join(context)
    logger.debug(f"Context prepared for response generation: {context_str}")

    # Log the first few messages of the context for debugging
    logger.debug(f"First few context messages: {context[:3]}")

    logger.info("Calling LLama for response generation")
    response = llama(system_prompt=system_prompt, prompt=f"Context:\n{context_str}\nUser: {new_message}\nAI:")
    # Clean the response if necessary
    response = response.strip()
    logger.info(f"Response generated: {response}")

    # Store the generated response in the memory manager
    ai_message_id = memory_manager.add_message_to_store(response, conversation_id, user_id, is_user_message=False)
    logger.debug(f"Added AI response with ID: {ai_message_id}")
    return response


def speak_text(text, language):
    logger.info(f"Converting text to speech. Language: {language}")
    logger.debug(f"Text to convert: {text}")

    tts = gTTS(text, lang=language)
    audio_file = "output/" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".mp3"
    tts.save(audio_file)
    logger.info(f"Audio file saved: {audio_file}")

    return audio_file

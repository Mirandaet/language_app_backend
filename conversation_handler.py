# Import necessary libraries
# import pyaudio
# import wave
import whisper
# import pyttsx3
import logging
from openai import OpenAI
import os
# from gtts import gTTS
from lms.ollama import llama
# import threadingp
import os
from queue import Queue
# from lms.romanji import japanese_to_romanji
# from frontend import main as frontend_main, ConversationApp
from datetime import datetime
import json
import torch, librosa
import io
import numpy as np
import soundfile as sf

logging.basicConfig(filename="start_japanese_logs.txt", level=logging.DEBUG, format=" %(asctime)s - %(levelname)s - %(message)s")

# Function to transcribe audio using Whisper
def transcribe_audio(audio_data):
    logging.debug("start of transcribe audio")
    model = whisper.load_model("base")
    
    # Convert audio data to numpy array
    with io.BytesIO(audio_data) as audio_file:
        audio_array, sample_rate = sf.read(audio_file)
    
    # Ensure audio is mono
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    
    # Convert to float32 (which is what Whisper expects)
    audio_array = audio_array.astype(np.float32)
    
    # Normalize the audio
    audio_array = audio_array / np.max(np.abs(audio_array))
    
    result = model.transcribe(audio_array)
    logging.debug(f"end of transcribe audio, returning {result['text']}")
    return result
# Function to generate a response using GPT-4
def generate_response(prompt, language):
    logging.debug("start of generate response")
    logging.info(f"prompt = {prompt}")

    with open("languages_desc.json", "r") as languages_file:
        languages = json.load(languages_file)

    system_prompt = languages[language][0]
    response = llama(system_prompt = system_prompt, prompt = prompt)

    response = response.removeprefix("AI")
    response = response.removeprefix(":")

    return response

def speak_text(text, language):
    tts = gTTS(text, lang=language)
    audio_file = "output/" + datetime.now().strftime("%Y%m%d-%H%M%S") +".mp3"
    tts.save(audio_file)

    return audio_file
    

# Main function to integrate all components


# def handle_conversation():
#     audio_filename = "output.wav"
#     # exit_keywords = ["exit", "quit", "終了"]
#     conversation_history = []

#     transcribed_text = transcribe_audio(audio_filename)
#     print(f"Transcribed Text: {transcribed_text}")

#     # Add the user's input to the conversation history
#     conversation_history.append(f"User: {transcribed_text}")

#     response = generate_response("  ".join(conversation_history))
#     print(f"Response: {response}")

#     # Add the AI's response to the conversation history
#     conversation_history.append(f"AI: {response}")

#     speak_text_japanese(response)

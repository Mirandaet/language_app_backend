# Import necessary libraries
import whisper
from openai import OpenAI
import os
from queue import Queue
from datetime import datetime
import json
import torch, librosa
import io
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer,  SetLogLevel
import wave
import logging
import argparse  # Add this import for command-line argument parsing

# Get the logger from the main script
logger = logging.getLogger(__name__)

def transcribe_audio(audio_data, language='EN', input_format='webm'):
    logger.info(f"Starting transcription for language: {language}, input format: {input_format}")
    
    vosk_models = {
        'EN': Model("vosk_models/vosk-model-en"),
        'ES': Model("vosk_models/vosk-model-es"),
        'FR': Model("vosk_models/vosk-model-fr"),
        'ZH': Model("vosk_models/vosk-model-zh"),
        'JP': Model("vosk_models/vosk-model-jp"),
        'KR': Model("vosk_models/vosk-model-kr"),
    }
    
    try:
        # Convert input to WAV in memory
        logger.debug(f"Converting {input_format} to WAV in memory")
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format=input_format)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
        
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Analyze audio properties
        samples = np.frombuffer(wav_io.getvalue(), dtype=np.int16)
        duration = len(samples) / 16000  # 16000 is the sample rate
        max_amplitude = np.max(np.abs(samples))
        rms = np.sqrt(np.mean(samples.astype(np.float32)**2))

        logger.info(f"Audio duration: {duration:.2f} seconds")
        logger.info(f"Max amplitude: {max_amplitude:.4f}")
        logger.info(f"RMS value: {rms:.4f}")

        if duration < 0.5:
            logger.warning("Audio is too short (less than 0.5 seconds)")
            return {"result": "Audio is too short"}

        if max_amplitude < 100:  # Adjusted threshold for int16 samples
            logger.warning("Audio is too quiet (max amplitude < 100)")
            return {"result": "Audio is too quiet"}

        # Prepare Vosk recognizer
        logger.debug(f"Initializing Vosk recognizer for {language}")
        model = vosk_models[language]
        rec = KaldiRecognizer(model, 16000)
        rec.SetWords(True)
        rec.SetPartialWords(True)

        # Process audio
        logger.debug("Processing audio")
        accumulated_text = ""
        wav_io.seek(0)
        while True:
            data = wav_io.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if "text" in result:
                    accumulated_text += result["text"] + " "
                    logger.debug(f"Partial result: {result['text']}")

        # Get final result
        final_result = json.loads(rec.FinalResult())
        logger.debug(f"Raw final result: {final_result}")

        if "text" in final_result and final_result["text"].strip():
            transcribed_text = final_result["text"].strip()
        else:
            transcribed_text = accumulated_text.strip()

        logger.info(f"Transcription result: {transcribed_text}")
        
        if not transcribed_text:
            logger.warning("Empty transcription result")
            return {"result": "No speech detected"}
        
        return {"result": transcribed_text}

    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}", exc_info=True)
        return {"error": str(e)}

# Function to generate a response using GPT-4
# def generate_response(prompt, language):
#     logger.debug("start of generate response")
#     logger.info(f"prompt = {prompt}")
#     with open("languages_desc.json", "r") as languages_file:
#         languages = json.load(languages_file)
#     system_prompt = languages[language][0]
#     response = llama(system_prompt=system_prompt, prompt=prompt)
#     response = response.removeprefix("AI")
#     response = response.removeprefix(":")
#     return response.strip()     

# ... [previous imports and code remain the same]

# Function to generate a response using GPT-4
def generate_response(prompt, language):
    logger.info(f"Starting generate_response function for language: {language}")
    logger.debug(f"Received prompt: {prompt}")
    
    try:
        logger.debug("Opening languages_desc.json file")
        with open("languages_desc.json", "r") as languages_file:
            languages = json.load(languages_file)
        logger.info(f"Successfully loaded language descriptions for {len(languages)} languages")
        
        if language not in languages:
            logger.warning(f"Language '{language}' not found in languages_desc.json")
            return "Error: Unsupported language"
        
        system_prompt = languages[language][0]
        logger.debug(f"System prompt for language {language}: {system_prompt}")
        
        logger.info("Calling llama function to generate response")
        response = llama(system_prompt=system_prompt, prompt=prompt)
        logger.debug(f"Raw response from llama: {response}")
        
        # Clean up the response
        response = response.removeprefix("AI")
        response = response.removeprefix(":")
        cleaned_response = response.strip()
        
        logger.info(f"Generated response (first 50 chars): {cleaned_response[:50]}...")
        logger.debug(f"Full cleaned response: {cleaned_response}")
        
        return cleaned_response
    except FileNotFoundError:
        logger.error("languages_desc.json file not found", exc_info=True)
        return "Error: Language description file not found"
    except json.JSONDecodeError:
        logger.error("Error decoding languages_desc.json", exc_info=True)
        return "Error: Invalid language description file"
    except Exception as e:
        logger.error(f"Unexpected error in generate_response: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

# ... [rest of the code remains the same]


def speak_text(text, language):
    tts = gTTS(text, lang=language)
    audio_file = "output/" + datetime.now().strftime("%Y%m%d-%H%M%S") +".mp3"
    tts.save(audio_file)

    return audio_file
import logging
import warnings
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import List
from conversation_handler import transcribe_audio, speak_text, generate_response
from models import GenerateRequest
from meloTTS import MeloTTSGenerator
import shutil
import numpy as np
import io
import json
import soundfile as sf
import re
from log_config import setup_logger
import io
from fastapi import HTTPException, Form
import soundfile as sf
from logging.handlers import RotatingFileHandler
from sqlalchemy.orm import Session
from datetime import datetime
from memory_manager import MemoryManager

# Initialize the conversation handler

logger = setup_logger()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.load")

app = FastAPI()

# Initialize MeloTTSGenerator
logger.info("Initializing MeloTTSGenerator")
tts_generator = MeloTTSGenerator()

UPLOAD_DIR = "uploaded_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Ensure required directories exist
logger.info("Ensuring required directories exist")
for directory in ['public', 'uploads']:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Created or verified existence of '{directory}' directory")

# Serve static files
logger.info("Mounting static file directories")
app.mount("/public", StaticFiles(directory="public"), name="public")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Define allowed origins for CORS
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:8000"
]

# Add CORS middleware
logger.info("Adding CORS middleware")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

memory_manager = MemoryManager(window_size=10)

# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP exception: {str(exc)}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)}
    )

# Endpoints

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("English")):
    logger.info(f"Received transcription request for file: {file.filename}")
    try:
        # Read the file content directly into memory
        audio_data = await file.read()
        logger.info("File read successfully, starting transcription")
        result = transcribe_audio(audio_data, language=language)
        logger.info("Transcription completed successfully")
        return result["text"]
    except Exception as e:
        logger.error(f"An error occurred during transcription: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    language: str = Form(...),
    speaker_id: int = Form(None)
):
    logger.info(f"Received TTS request for text: '{text}' in language: {language}, speaker ID: {speaker_id}")
    try:
        logger.debug("Generating audio using MeloTTSGenerator")
        language_code = "EN"

        with open("languages_desc.json", "r") as f:
            languages_data = json.load(f)
            language_code = languages_data[language][1].upper()
            if speaker_id is None:
                speaker_id = languages_data[language][2]
        
        audio = tts_generator.generate_audio(text, speaker_id, language_code)
        
        if audio is not None:
            logger.debug("Audio generated successfully, preparing response")
            buffer = io.BytesIO()
            sf.write(buffer, audio, tts_generator.sampling_rate, format='wav')
            buffer.seek(0)
            
            logger.info("Returning audio as StreamingResponse")
            return StreamingResponse(buffer, media_type="audio/wav", headers={
                'Content-Disposition': 'attachment; filename="audio.wav"'
            })
        else:
            logger.error("Failed to generate audio")
            raise HTTPException(status_code=500, detail="Failed to generate audio")
    except Exception as e:
        logger.error(f"An error occurred during TTS: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available_voices")
async def get_available_voices():
    try:
        # This assumes your tts_generator has a method to get available voices
        voices = tts_generator.get_available_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"An error occurred while fetching available voices: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate_chatbot_response(request: GenerateRequest):
    logger.info(f"Received generate request for user {request.user_id}, conversation {request.conversation_id}")
    try:
        # Generate the response from the chatbot
        chatbot_response = generate_response(
            request.message, 
            request.conversation_id, 
            request.user_id, 
            request.language
        )
        logger.info("Response generated successfully")
        return {"response": chatbot_response}
    except Exception as e:
        logger.error(f"An error occurred during response generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


    
@app.get("/fetch_languages")
async def fetch_languages():
    logger.info("Received request to fetch languages")
    try:
        # Log the current working directory
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Log the full path of the languages.json file
        languages_file_path = os.path.join(os.getcwd(), "languages.json")
        logger.info(f"Attempting to open file: {languages_file_path}")
        
        with open(languages_file_path, "r") as languages_file:
            languages = json.load(languages_file)
        
        logger.info(f"Successfully fetched {len(languages)} languages")
        return languages
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"languages.json file not found: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON in languages.json: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error fetching languages: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch languages: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
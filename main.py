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
from models import ChatHistory
from meloTTS import MeloTTSGenerator
import shutil
import numpy as np
import io
import json
import soundfile as sf
import re
import io
from fastapi import HTTPException, Form
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.load")

app = FastAPI()

# Initialize MeloTTSGenerator
logger.info("Initializing MeloTTSGenerator")
tts_generator = MeloTTSGenerator()

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
async def transcribe(file: UploadFile = File(...)):
    logger.info(f"Received transcription request for file: {file.filename}")
    try:
        # Read the file content directly into memory
        audio_data = await file.read()
        logger.info("File read successfully, starting transcription")
        result = transcribe_audio(audio_data)
        logger.info("Transcription completed successfully")
        return {"result": result["text"]}
    except Exception as e:
        logger.error(f"An error occurred during transcription: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    language: str = Form(...),
    speaker_id: int = Form(None)  # New parameter for speaker ID
):
    logger.info(f"Received TTS request for text: '{text}' in language: {language}, speaker ID: {speaker_id}")
    try:
        logger.debug("Generating audio using MeloTTSGenerator")
        
        # If speaker_id is None, it will use the default voice
        audio = tts_generator.generate_audio(text, speaker_id=speaker_id)
        
        if audio is not None:
            logger.debug("Audio generated successfully, preparing response")
            buffer = io.BytesIO()
            logger.debug(f"Writing audio to buffer with sampling rate: {tts_generator.sampling_rate}")
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
async def generate(data: ChatHistory):
    logger.info("Received request to generate response")
    logger.debug(f"Chat history length: {len(data.chat_history)}")
    chat_history = ""
    for message in data.chat_history:
        chat_history += (message["role"] + ":" + message["message"])
        chat_history += "\n"
    logger.debug("Generating response")
    response = generate_response(chat_history, data.language)
    logger.info("Response generated successfully")
    return response

@app.get("/fetch_languages")
async def fetch_languages():
    logger.info("Received request to fetch languages")
    try:
        with open("languages.json", "r") as languages_file:
            languages = json.load(languages_file)
        logger.info(f"Successfully fetched {len(languages)} languages")
        return languages
    except Exception as e:
        logger.error(f"Error fetching languages: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch languages")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
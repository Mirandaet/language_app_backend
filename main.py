from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydub import AudioSegment
from conversation_handler import transcribe_audio, speak_text, generate_response, text_to_speech_waveform
from models import ChatHistory
import shutil
import os
import numpy as np
import io
import json
import logging


app = FastAPI()

# Serve static files
app.mount("/public", StaticFiles(directory="public"), name="public")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Define allowed origins
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:8000"
    # Add more origins as needed
]

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins as listed above
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Allow all common HTTP methods
    allow_headers=["*"],  # Allows all HTTP headers
)


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        file_location = f"uploads/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ensure this function is correctly defined and working
        result = transcribe_audio(file_location)
        return {result}
    except Exception as e:
        print(f"An error occurred: {str(e)}")  # Logs the error to the console
        return {"error": str(e)}, 500  # Returns the error and a 500 status


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    language: str = Form(...)
):
    language_code = "en"
    try:
        with open("languages_desc.json") as file:
            language_code = json.load(file)[language][1]
    except KeyError:
        pass
    # Generate speech audio from text
    file_name = text_to_speech_waveform(text, language_code)
    return FileResponse(file_name, media_type="audio/wav", filename=file_name)


@app.post("/generate")
async def generate(data: ChatHistory):
    chat_history = ""
    for message in data.chat_history:
        chat_history += (message["role"] + ":" + message["message"])
        chat_history += "\n"
    response = generate_response(chat_history, data.language)
    return response


@app.get("/fetch_languages")
async def fetch_languages():
    with open("languages.json", "r") as languages_file:
        languages = json.load(languages_file)
    return languages


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

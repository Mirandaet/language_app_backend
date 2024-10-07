# Import necessary libraries
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
from pydub import AudioSegment
from pydub.effects import normalize
import whisper

# Get the logger from the main script
logger = logging.getLogger(__name__)





def transcribe_audio(content, language, input_format='webm'):
    # Load the Whisper model
    model = whisper.load_model("small")

    # Convert the byte content to an audio segment using pydub
    audio_segment = AudioSegment.from_file(io.BytesIO(content), format=input_format)

    # Resample to 16000 Hz as required by Whisper
    audio_segment = audio_segment.set_frame_rate(16000)

    # Convert to mono if stereo
    if audio_segment.channels == 2:
        audio_segment = audio_segment.set_channels(1)

    # Convert the audio segment to a numpy array
    samples = np.array(audio_segment.get_array_of_samples()).flatten().astype(np.float32)

    # Normalize
    samples /= np.max(np.abs(samples))

    # Transcribe the audio
    result = model.transcribe(samples, language=language)
    
    logger.info(f"Transcription result: {result['text']}")
    
    return result


# def transcribe_audio(audio_data, language='EN', input_format='webm'):
#     logger.info(f"Starting transcription for language: {language}, input format: {input_format}")
    
#     vosk_models = {
#         'EN': Model("vosk_models/vosk-model-en"),
#         'ES': Model("vosk_models/vosk-model-es"),
#         'FR': Model("vosk_models/vosk-model-fr"),
#         'ZH': Model("vosk_models/vosk-model-zh"),
#         'JP': Model("vosk_models/vosk-model-jp"),
#         'KR': Model("vosk_models/vosk-model-kr"),
#     }
    
#     try:
#         # Convert input to WAV in memory
#         logger.debug(f"Converting {input_format} to WAV in memory")
#         audio = AudioSegment.from_file(io.BytesIO(audio_data), format=input_format)
#         audio = audio.set_channels(1)  # Convert to mono
#         audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
        
#         # Normalize audio
#         audio = normalize(audio)
        
#         # Trim silence
#         audio = audio.strip_silence(silence_len=1000, silence_thresh=-40)
        
#         wav_io = io.BytesIO()
#         audio.export(wav_io, format="wav")
#         wav_io.seek(0)

        
#         wav_path = "processed_audio.wav"
#         audio.export(wav_path, format="wav")
#         logger.info(f"Saved processed audio to {wav_path}")

#         # Analyze audio properties
#         samples = np.frombuffer(wav_io.getvalue(), dtype=np.int16)
#         duration = len(samples) / 16000  # 16000 is the sample rate
#         max_amplitude = np.max(np.abs(samples))
#         rms = np.sqrt(np.mean(samples.astype(np.float32)**2))

#         logger.info(f"Audio duration: {duration:.2f} seconds")
#         logger.info(f"Max amplitude: {max_amplitude:.4f}")
#         logger.info(f"RMS value: {rms:.4f}")

#         if duration < 0.5:
#             logger.warning("Audio is too short (less than 0.5 seconds)")
#             return {"result": "Audio is too short"}

#         if max_amplitude < 100:  # Adjusted threshold for int16 samples
#             logger.warning("Audio is too quiet (max amplitude < 100)")
#             return {"result": "Audio is too quiet"}

#         # Prepare Vosk recognizer
#         logger.debug(f"Initializing Vosk recognizer for {language}")
#         model = vosk_models[language]
#         rec = KaldiRecognizer(model, 16000)
#         rec.SetWords(True)
#         rec.SetPartialWords(True)

#         # Process audio
#         logger.debug("Processing audio")
#         accumulated_text = ""
#         wav_io.seek(0)
#         while True:
#             data = wav_io.read(4000)
#             if len(data) == 0:
#                 break
#             if rec.AcceptWaveform(data):
#                 result = json.loads(rec.Result())
#                 if "text" in result:
#                     accumulated_text += result["text"] + " "
#                     logger.debug(f"Partial result: {result['text']}")

#         # Get final result
#         final_result = json.loads(rec.FinalResult())
#         logger.debug(f"Raw final result: {final_result}")

#         if "text" in final_result and final_result["text"].strip():
#             transcribed_text = final_result["text"].strip()
#         else:
#             transcribed_text = accumulated_text.strip()

#         logger.info(f"Transcription result: {transcribed_text}")
        
#         if not transcribed_text:
#             logger.warning("Empty transcription result")
#             return {"result": "No speech detected"}
        
#         return {"result": transcribed_text}

#     except Exception as e:
#         logger.error(f"Error in transcribe_audio: {str(e)}", exc_info=True)
#         return {"error": str(e)}

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

import argparse
import logging
import os
from pydub import AudioSegment
import io

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio file using Vosk")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("--language", default="EN", choices=["EN", "ES", "FR", "ZH", "JP", "KR"],
                        help="Language of the audio (default: EN)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        # Get the file extension
        _, file_extension = os.path.splitext(args.audio_file)
        input_format = file_extension[1:].lower()  # Remove the dot from the extension

        if input_format not in ['mp3', 'wav']:
            raise ValueError(f"Unsupported file format: {input_format}")

        # Read the audio file
        with open(args.audio_file, "rb") as audio_file:
            audio_data = audio_file.read()

        # If it's a WAV file, we need to ensure it's in the correct format
        if input_format == 'wav':
            audio = AudioSegment.from_wav(io.BytesIO(audio_data))
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            audio_data = wav_io.getvalue()

        result = transcribe_audio(audio_data, language=args.language, input_format=input_format)

        if "text" in result:
            print(f"Transcription result: {result['text']}")
        elif "error" in result:
            print(f"Error during transcription: {result['error']}")
        else:
            print("Unexpected result format")

    except FileNotFoundError:
        print(f"Error: The file '{args.audio_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
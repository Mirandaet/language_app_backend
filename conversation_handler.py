# Import necessary libraries
# import pyaudio
# import wave
# import whisper
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
from frontend import main as frontend_main, ConversationApp
from datetime import datetime
import json
import torch, librosa

logging.basicConfig(filename="start_japanese_logs.txt", level=logging.DEBUG, format=" %(asctime)s - %(levelname)s - %(message)s")

# Function to transcribe audio using Whisper
def transcribe_audio(filename):
    logging.debug("start of transcribe audio")
    logging.info(f"filename = {filename}")
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    logging.debug(f"end of transcribe audio, returning {result["text"]}")
    return result["text"]


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

mars5, config_class = torch.hub.load('Camb-ai/mars5-tts', 'mars5_english', trust_repo=True)
# The `mars5` contains the AR and NAR model, as well as inference code.
# The `config_class` contains tunable inference config settings like temperature.

directory = "../voices/Vol 1"

# Initialize a list to hold the reference audio
reference_audios = []

# Load all WAV files from the directory
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(directory, filename)
        wav, sr = librosa.load(file_path, sr=mars5.sr, mono=True)
        reference_audios.append(torch.from_numpy(wav))

# If you want to concatenate all the reference audio into one tensor
if reference_audios:
    combined_reference_audio = torch.cat(reference_audios)

# Combine transcripts from all reference audio files
reference_transcripts = []
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        # Assuming the transcript can be derived from the filename, modify as needed
        transcript = filename.replace('.wav', '')  # or any method to get the transcript
        reference_transcripts.append(transcript)

# Combine into a single string
ref_transcript = ' '.join(reference_transcripts)


# Pick whether you want a deep or shallow clone. Set to False if you don't know prompt transcript or want fast inference. Set to True if you know transcript and want highest quality.
deep_clone = True
# Below you can tune other inference settings, like top_k, temperature, top_p, etc...
cfg = config_class(deep_clone=deep_clone, rep_penalty_window=100,
                      top_k=100, temperature=0.7, freq_penalty=3)

ar_codes, output_audio = mars5.tts("The quick brown rat.", combined_reference_audio, ref_transcript,cfg=cfg)
# output_audio is (T,) shape float tensor corresponding to the 24kHz output audio.
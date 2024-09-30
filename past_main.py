# Import necessary libraries
import pyaudio
import wave
import whisper
import pyttsx3
import logging
from openai import OpenAI
import os
from gtts import gTTS
import pygame
from lms.ollama import llama
import threading
import os
from queue import Queue
# from romaji.romanji import japanese_to_romanji
from frontend import main as frontend_main, ConversationApp

logging.basicConfig(filename="start_japanese_logs.txt", level=logging.DEBUG, format=" %(asctime)s - %(levelname)s - %(message)s")

# Function to record audio from the microphone
is_recording = False

class ConversationHandler:
    def __init__(self, command_queue, update_callback):
        self.command_queue = command_queue
        self.update_callback = update_callback
        self.is_recording = False
        self.conversation_history = []

    def record_audio(self, filename):
        global is_recording
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 1
        fs = 44100

        p = pyaudio.PyAudio()

        print('Recording...')

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []

        while self.is_recording:
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        print('Finished recording')

        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

    def start_recording(self):
        global is_recording
        self.is_recording = True
        threading.Thread(target=self.record_audio, args=("output/output.wav",)).start()

    def stop_recording(self):
        self.is_recording = False

    # Function to transcribe audio using Whisper
    def transcribe_audio(self,  ):
        logging.debug("start of transcribe audio")
        logging.info(f"filename = {filename}")
        model = whisper.load_model("base")
        result = model.transcribe(filename)
        logging.debug(f"end of transcribe audio, returning {result["text"]}")
        return result["text"]


    # Function to generate a response using GPT-4
    def generate_response(self, prompt):
        logging.debug("start of generate response")
        logging.info(f"prompt = {prompt}")
        system_prompt = "You are a helpful language assistant talking to someone only speaking in japanese, you only answer in kana and kanji characters" 
        response = llama(system_prompt = system_prompt, prompt = prompt)
        return response

    def speak_text_japanese(self, text):
        tts = gTTS(text, lang='ja')
        audio_file = "output.mp3"
        tts.save(audio_file)

        # Initialize pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.unload()
        pygame.mixer.quit()
        
        os.remove(audio_file)

    # Main function to integrate all components


    def handle_conversation(self):
        audio_filename = "output.wav"
        exit_keywords = ["exit", "quit", "終了"]

        while True:
            command = self.command_queue.get()
            if command == "start":
                self.start_recording()
            elif command == "stop":
                self.stop_recording()
                transcribed_text = self.transcribe_audio(audio_filename)
                print(f"Transcribed Text: {transcribed_text}")

                # Add the user's input to the conversation history
                self.conversation_history.append(f"User: {transcribed_text}")

                response = self.generate_response("  ".join(self.conversation_history))
                print(f"Response: {response}")

                # Add the AI's response to the conversation history
                self.conversation_history.append(f"AI: {response}")

                self.speak_text_japanese(response)

                self.update_callback



def main():
    command_queue = Queue()
    update_queue = Queue()

    handler = ConversationHandler(command_queue, update_queue)

    frontend_thread = threading.Thread(target=frontend_main, args=(command_queue, update_queue))
    frontend_thread.start()

    handler.handle_conversation()

if __name__ == "__main__":
    main()

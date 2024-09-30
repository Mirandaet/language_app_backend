from gtts import gTTS
import pygame
import os

def speak_text_japanese(text):
    tts = gTTS(text, lang='ja')
    audio_file = "output.mp3"
    tts.save(audio_file)

    # Initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    os.remove(audio_file)

# Example usage
speak_text_japanese("こんにちは、元気ですか？")

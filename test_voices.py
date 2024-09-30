import pyttsx3

def speak_text_japanese(text):
    engine = pyttsx3.init()

    # List available voices
    voices = engine.getProperty('voices')
    for voice in voices:
        print(f"Voice: {voice.name}, ID: {voice.id}, Languages: {voice.languages}")

    # Set the voice to a Japanese voice if available
    japanese_voice = None
    for voice in voices:
        if 'Japanese' in voice.name:
            japanese_voice = voice
            break

    if japanese_voice:
        engine.setProperty('voice', japanese_voice.id)
        engine.say(text)
        engine.runAndWait()
    else:
        print("No Japanese voice available.")

# Example usage
speak_text_japanese("こんにちは、元気ですか？")

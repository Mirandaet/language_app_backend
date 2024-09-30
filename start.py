# Import necessary libraries
import pyaudio
import wave
import whisper
import openai
import pyttsx3
import logging
from openai import OpenAI
import os

logging.basicConfig(filename="start_logs.txt", level=logging.DEBUG, format=" %(asctime)s - %(levelname)s - %(message)s")

# Function to record audio from the microphone
def record_audio(filename, record_seconds=5):
    logging.debug("start of record audio")
    logging.info(f"filename = {filename}, record_seconds = {record_seconds}")
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for the given number of seconds
    for _ in range(0, int(fs / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    logging.debug("end of record audio")

# Function to transcribe audio using Whisper
def transcribe_audio(filename):
    logging.debug("start of transcribe audio")
    logging.info(f"filename = {filename}")
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    logging.debug(f"end of transcribe audio, returning {result["text"]}")
    return result["text"]


# Function to generate a response using GPT-4
def generate_response(prompt):
    logging.debug("start of generate response")
    logging.info(f"prompt = {prompt}")
    client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
    )

    system_prompt = "You are having a conversation with a friend"

    completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    )
    logging.debug(f"end of generate response, returning {completion.choices[0].message}")
    return completion.choices[0].message.content


    # response = openai.completions.create(
    #     model="text-davinci-003",
    #     prompt=prompt,
    #     max_tokens=150
    # )
    # return response.choices[0].text.strip()

# Function to convert text to speech using pyttsx3
def speak_text(text):
    logging.debug("start of speak text")
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    logging.debug("end of speak text")

# Main function to integrate all components
def main():
    logging.debug("start of main function")
    audio_filename = "/Users/Miranda/Documents/side_project/Japanese_conversations/output.wav"
    record_audio(audio_filename)
    transcribed_text = transcribe_audio(audio_filename)
    print(f"Transcribed Text: {transcribed_text}")
    response = generate_response(transcribed_text)
    print(f"Response: {response}")
    speak_text(response)
    logging.debug("end of main function")

if __name__ == "__main__":
    main()

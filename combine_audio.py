from pydub import AudioSegment

# Load your audio files
audio_files = ["file1.wav", "file2.wav", "file3.wav"]  # Replace with your actual file paths

# Initialize an empty audio segment
combined = AudioSegment.empty()

# Concatenate all audio files
for file in audio_files:
    audio_segment = AudioSegment.from_wav(file)
    combined += audio_segment

# Export the combined audio file
combined.export("combined_audio.wav", format="wav")

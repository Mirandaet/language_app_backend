import sys
import os
import torch
from typing import Optional
import numpy as np

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class MeloTTSGenerator:
    def __init__(self, language: str = "EN", device: str = "cpu"):
        from MeloTTS.melo.api import TTS
        self.tts = TTS(language=language, device=device)
        self.sampling_rate = self.tts.hps.data.sampling_rate
        self.available_voices = self._get_available_voices()

    def generate_audio(self, text: str, speaker_id: int = 0) -> Optional[np.ndarray]:
        try:
            audio = self.tts.tts_to_file(text, speaker_id)
            return audio
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def _get_available_voices(self):
        # This method should return a list of available voices
        # The exact implementation depends on how MeloTTS stores this information
        # This is a placeholder implementation
        return list(range(self.tts.hps.data.n_speakers))
    
    def get_available_voices(self):
        return self.available_voices


# Example usage
if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current directory: {current_dir}")
    print(f"Parent directory: {parent_dir}")
    print(f"Python path: {sys.path}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch installation path: {torch.__file__}")

    # Initialize the generator once
    generator = MeloTTSGenerator()

    # Generate audio
    text = "Hello, this is a test of the MeloTTS system."
    audio = generator.generate_audio(text)
    
    if audio is not None:
        print(f"Audio generated successfully. Shape: {audio.shape}, Sampling rate: {generator.sampling_rate}")
        
        # Save the audio
        import soundfile as sf
        output_path = os.path.join(current_dir, "output.wav")
        sf.write(output_path, audio, generator.sampling_rate)
        print(f"Audio saved to: {output_path}")
    else:
        print("Failed to generate audio.")
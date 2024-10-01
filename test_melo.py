import sys
import os
import torch
from typing import Optional
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class MeloTTSGenerator:
    def __init__(self, language: str = "EN", device: str = "cpu"):
        logger.info(f"Initializing MeloTTSGenerator with language: {language}, device: {device}")
        from MeloTTS.melo.api import TTS
        self.tts = TTS(language=language, device=device)
        self.sampling_rate = self.tts.hps.data.sampling_rate
        logger.info(f"MeloTTSGenerator initialized. Sampling rate: {self.sampling_rate}")

    def generate_audio(self, text: str, speaker_id: int = 0) -> Optional[np.ndarray]:
        try:
            logger.info(f"Generating audio for text: '{text}' with speaker_id: {speaker_id}")
            audio = self.tts.tts_to_file(text, speaker_id)
            logger.info(f"Audio generated successfully. Shape: {audio.shape}")
            return audio
        except Exception as e:
            logger.error(f"An error occurred during audio generation: {e}")
            logger.exception("Exception details:")
            return None

# Example usage
if __name__ == "__main__":
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current directory: {current_dir}")
    logger.info(f"Parent directory: {parent_dir}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"Torch installation path: {torch.__file__}")

    # Initialize the generator once
    generator = MeloTTSGenerator()

    # Generate audio
    text = "Hello, this is a test of the MeloTTS system."
    audio = generator.generate_audio(text)
    
    if audio is not None:
        logger.info(f"Audio generated successfully. Shape: {audio.shape}, Sampling rate: {generator.sampling_rate}")
        
        # Save the audio
        import soundfile as sf
        output_path = os.path.join(current_dir, "output.wav")
        sf.write(output_path, audio, generator.sampling_rate)
        logger.info(f"Audio saved to: {output_path}")
    else:
        logger.error("Failed to generate audio.")
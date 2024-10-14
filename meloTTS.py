import sys
import os
import torch
from typing import Optional
import numpy as np
import logging

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from MeloTTS.melo.api import TTS

logger = logging.getLogger(__name__)

class MeloTTSGenerator:
    def __init__(self, language: str = "EN", device: str = "cpu"):
        logger.info(f"Initializing MeloTTSGenerator with language: {language} and device: {device}")
        from MeloTTS.melo.api import TTS
        self.tts_models = {}
        self.current_language = language
        self.device = device
        self.load_tts_model(language)
        self.sampling_rate = self.tts_models[language].hps.data.sampling_rate
        self.available_voices = self._get_available_voices()
        logger.debug(f"Initialization complete. Available voices: {len(self.available_voices)}")

    def load_tts_model(self, language: str):
        logger.info(f"Loading TTS model for language: {language}")
        if language not in self.tts_models:
            try:
                self.tts_models[language] = TTS(language=language, device=self.device)
                logger.info(f"Successfully loaded TTS model for language: {language}")
            except Exception as e:
                logger.error(f"Failed to load TTS model for language {language}. Error: {str(e)}")
                raise
        self.current_language = language

    def generate_audio(self, text: str, speaker_id: int = 0, language: str = None) -> Optional[np.ndarray]:
        logger.info(f"Generating audio for text: '{text}', speaker_id: {speaker_id}, language: {language}")
        try:
            if language and language != self.current_language:
                logger.info(f"Switching language from {self.current_language} to {language}")
                self.load_tts_model(language)
           
            audio = self.tts_models[self.current_language].tts_to_file(text, speaker_id)
            logger.info(f"Successfully generated audio. Shape: {audio.shape}")
            return audio
        except Exception as e:
            logger.error(f"An error occurred while generating audio: {str(e)}")
            logger.exception("Traceback:")
            return None
       
    def _get_available_voices(self):
        voices = list(range(self.tts_models[self.current_language].hps.data.n_speakers))
        logger.debug(f"Available voices for language {self.current_language}: {voices}")
        return voices
   
    def get_available_voices(self):
        logger.debug(f"Returning available voices: {self.available_voices}")
        return self.available_voices

    def get_available_languages(self):
        languages = list(self.tts_models.keys())
        logger.debug(f"Available languages: {languages}")
        return languages
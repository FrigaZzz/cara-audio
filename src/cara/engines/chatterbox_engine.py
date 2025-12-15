"""
Chatterbox TTS Engine wrapper.

Provides a unified interface for all Chatterbox model variants:
- multilingual: 23 languages including Italian
- turbo: English only, faster inference
- original: English, supports emotion tags
"""

import logging
from pathlib import Path
from typing import Literal

import torch

logger = logging.getLogger(__name__)


class ChatterboxEngine:
    """
    Unified Chatterbox TTS engine supporting all variants.
    
    MIT Licensed - fully commercial use allowed.
    """
    
    def __init__(
        self,
        variant: Literal["multilingual", "turbo", "original"] = "multilingual",
        device: str = "cuda",
    ):
        """
        Initialize Chatterbox engine.
        
        Args:
            variant: Model variant - "multilingual" for Italian support
            device: "cuda" or "cpu"
        """
        self.variant = variant
        self.device = device
        self._model = None
        self._sample_rate: int = 24000  # Default, will be updated after load
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the appropriate Chatterbox model."""
        logger.info(f"Loading Chatterbox {self.variant} model on {self.device}...")
        
        if self.variant == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        elif self.variant == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)
        else:  # original
            from chatterbox.tts import ChatterboxTTS
            self._model = ChatterboxTTS.from_pretrained(device=self.device)
        
        self._sample_rate = self._model.sr
        logger.info(f"✅ Chatterbox {self.variant} loaded (sample_rate={self._sample_rate})")
    
    @property
    def sample_rate(self) -> int:
        """Get the model's output sample rate."""
        return self._sample_rate
    
    def generate(
        self,
        text: str,
        language: str = "it",
        speaker_wav: str | Path | None = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Generate speech audio from text with advanced parameters.
        
        Args:
            text: Text to synthesize (can include emotion tags like [laugh], [sigh])
            language: ISO 639-1 language code (e.g., "it" for Italian)
            speaker_wav: Optional path to speaker reference audio for voice cloning
            exaggeration: Emotion intensity (0.3-0.4 neutral, 0.5 balanced, 0.7+ expressive)
            temperature: Creativity/randomness (0.4-0.6 consistent, 0.8 balanced, 1.0+ creative)
            cfg_weight: Voice fidelity to reference (higher = more faithful)
            seed: Random seed for reproducible generations
            
        Returns:
            Audio tensor (1D or 2D)
        """
        speaker_path = str(speaker_wav) if speaker_wav else None
        
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Build generation kwargs based on model variant
        gen_kwargs = {}
        
        if self.variant == "multilingual":
            wav = self._model.generate(
                text,
                language_id=language,
                audio_prompt_path=speaker_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
        elif self.variant == "turbo":
            # Turbo requires a reference audio for voice cloning
            if not speaker_path:
                logger.warning("Turbo model works best with a reference audio prompt")
            wav = self._model.generate(
                text, 
                audio_prompt_path=speaker_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
        else:  # original
            wav = self._model.generate(
                text, 
                audio_prompt_path=speaker_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
        
        return wav
    
    def generate_with_emotion(
        self,
        text: str,
        language: str = "it",
        speaker_wav: str | Path | None = None,
    ) -> torch.Tensor:
        """
        Generate speech with paralinguistic tags (emotion).
        
        The text can include tags like [laugh], [chuckle], etc.
        Works best with turbo/original models.
        
        Example:
            text = "Hi there [chuckle], how are you today?"
        """
        return self.generate(text, language, speaker_wav)

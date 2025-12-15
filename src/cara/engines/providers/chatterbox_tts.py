"""
Chatterbox TTS Provider.

Wraps the Chatterbox TTS engine with the TTSProvider interface.
"""

import io
import logging
import struct
import wave
from pathlib import Path
from typing import AsyncIterator, Literal

import torch

from ..providers.tts_base import TTSProvider, TTSResult

logger = logging.getLogger(__name__)


class ChatterboxTTSProvider(TTSProvider):
    """Chatterbox TTS provider with support for multilingual and emotion tags."""
    
    def __init__(
        self,
        variant: Literal["multilingual", "turbo", "original"] = "multilingual",
        device: str = "cuda",
    ):
        self.variant = variant
        self.device = device
        self._model = None
        self._sample_rate = 24000
    
    @property
    def name(self) -> str:
        return f"chatterbox-{self.variant}"
    
    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    
    @property
    def supports_streaming(self) -> bool:
        return False  # Chatterbox generates full audio
    
    @property
    def supports_voice_cloning(self) -> bool:
        return True
    
    async def load(self) -> None:
        """Load the Chatterbox model."""
        if self._model is not None:
            return
        
        logger.info(f"Loading Chatterbox {self.variant} model on {self.device}...")
        
        if self.variant == "multilingual":
            from chatterbox.tts import ChatterboxMultilingualTTS
            self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        elif self.variant == "turbo":
            from chatterbox.tts import ChatterboxTTSTurbo
            self._model = ChatterboxTTSTurbo.from_pretrained(device=self.device)
        else:
            from chatterbox.tts import ChatterboxTTS
            self._model = ChatterboxTTS.from_pretrained(device=self.device)
        
        self._sample_rate = self._model.sr
        logger.info(f"✅ Chatterbox {self.variant} loaded (sample_rate={self._sample_rate})")
    
    async def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Chatterbox model unloaded")
    
    async def generate(
        self,
        text: str,
        language: str = "en",
        voice_id: str | None = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        seed: int | None = None,
        speaker_wav: str | Path | None = None,
        **kwargs,
    ) -> TTSResult:
        """
        Generate speech from text using Chatterbox.
        
        Args:
            text: Text to synthesize (can include emotion tags for turbo)
            language: ISO 639-1 language code
            voice_id: Voice profile ID (maps to speaker wav)
            exaggeration: Emotion intensity (0.25-2.0)
            temperature: Creativity (0.05-5.0)
            cfg_weight: Voice fidelity (0.0-1.0)
            seed: Random seed for reproducibility
            speaker_wav: Path to speaker reference audio
        """
        if self._model is None:
            await self.load()
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        speaker_path = str(speaker_wav) if speaker_wav else None
        
        # Generate audio
        if self.variant == "multilingual":
            wav_tensor = self._model.generate(
                text,
                language_id=language,
                audio_prompt_path=speaker_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
        else:
            wav_tensor = self._model.generate(
                text,
                audio_prompt_path=speaker_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
        
        # Convert to bytes
        audio_np = wav_tensor.squeeze().cpu().numpy()
        audio_bytes = self._to_wav_bytes(audio_np)
        duration = len(audio_np) / self._sample_rate
        
        return TTSResult(
            audio=audio_bytes,
            sample_rate=self._sample_rate,
            duration_seconds=duration,
            format="wav",
        )
    
    def _to_wav_bytes(self, audio_np) -> bytes:
        """Convert numpy audio to WAV bytes."""
        import numpy as np
        
        # Normalize to int16
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()

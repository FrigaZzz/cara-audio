"""
TTS Provider Base Interface.

Abstract base class for all TTS providers (local and cloud).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator
from pathlib import Path


@dataclass
class TTSResult:
    """Result from TTS generation."""
    audio: bytes
    sample_rate: int
    duration_seconds: float
    format: str = "wav"


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'chatterbox', 'elevenlabs')."""
        pass
    
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming output."""
        pass
    
    @property
    def supports_voice_cloning(self) -> bool:
        """Whether this provider supports voice cloning."""
        return False
    
    @abstractmethod
    async def generate(
        self,
        text: str,
        language: str = "en",
        voice_id: str | None = None,
        **kwargs,
    ) -> TTSResult:
        """
        Generate speech from text.
        
        Args:
            text: Text to synthesize
            language: ISO 639-1 language code
            voice_id: Optional voice profile ID
            **kwargs: Provider-specific parameters
            
        Returns:
            TTSResult with audio bytes
        """
        pass
    
    async def stream(
        self,
        text: str,
        language: str = "en",
        voice_id: str | None = None,
        **kwargs,
    ) -> AsyncIterator[bytes]:
        """
        Stream speech generation (optional).
        
        Override if provider supports streaming.
        Default implementation generates full audio then yields it.
        """
        result = await self.generate(text, language, voice_id, **kwargs)
        yield result.audio
    
    async def load(self) -> None:
        """Load/initialize the provider (optional)."""
        pass
    
    async def unload(self) -> None:
        """Unload/cleanup the provider (optional)."""
        pass

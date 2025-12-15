"""
STT Provider Base Interface.

Abstract base class for all STT providers (local and cloud).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator
from pathlib import Path


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing."""
    start: float
    end: float
    text: str
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    """Result from STT transcription."""
    text: str
    language: str
    duration_seconds: float
    segments: list[TranscriptionSegment] = field(default_factory=list)


class STTProvider(ABC):
    """Abstract base class for STT providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'whisper', 'deepgram')."""
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming input."""
        pass
    
    @property
    def supports_language_detection(self) -> bool:
        """Whether this provider can auto-detect language."""
        return False
    
    @abstractmethod
    async def transcribe(
        self,
        audio: bytes | Path,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio bytes or path to audio file
            language: ISO 639-1 language code (None for auto-detect)
            **kwargs: Provider-specific parameters
            
        Returns:
            TranscriptionResult with text and segments
        """
        pass
    
    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[bytes],
        language: str | None = None,
        **kwargs,
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Stream transcription (optional).
        
        Override if provider supports streaming.
        Default implementation buffers all audio then transcribes.
        """
        # Buffer audio chunks
        chunks = []
        async for chunk in audio_stream:
            chunks.append(chunk)
        
        audio = b"".join(chunks)
        result = await self.transcribe(audio, language, **kwargs)
        yield result
    
    async def load(self) -> None:
        """Load/initialize the provider (optional)."""
        pass
    
    async def unload(self) -> None:
        """Unload/cleanup the provider (optional)."""
        pass

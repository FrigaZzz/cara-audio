"""
Whisper STT Provider.

Wraps faster-whisper with the STTProvider interface.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import AsyncIterator, Literal

from ..providers.stt_base import STTProvider, TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


class WhisperSTTProvider(STTProvider):
    """Faster-Whisper STT provider."""
    
    def __init__(
        self,
        model_size: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
    
    @property
    def name(self) -> str:
        return f"whisper-{self.model_size}"
    
    @property
    def supports_streaming(self) -> bool:
        return False  # Whisper processes full audio
    
    @property
    def supports_language_detection(self) -> bool:
        return True
    
    async def load(self) -> None:
        """Load the Whisper model."""
        if self._model is not None:
            return
        
        logger.info(f"Loading Whisper {self.model_size} on {self.device}...")
        
        from faster_whisper import WhisperModel
        
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        
        logger.info(f"✅ Whisper {self.model_size} loaded")
    
    async def unload(self) -> None:
        """Unload the model."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Whisper model unloaded")
    
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
            language: ISO 639-1 code (None for auto-detect)
        """
        if self._model is None:
            await self.load()
        
        # Handle bytes input
        if isinstance(audio, bytes):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                audio_path = f.name
            cleanup = True
        else:
            audio_path = str(audio)
            cleanup = False
        
        try:
            # Run transcription in executor to not block
            loop = asyncio.get_event_loop()
            
            def _transcribe():
                segments, info = self._model.transcribe(
                    audio_path,
                    language=language,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500},
                )
                return list(segments), info
            
            segments, info = await loop.run_in_executor(None, _transcribe)
            
            # Build result
            text_parts = []
            result_segments = []
            
            for seg in segments:
                text_parts.append(seg.text.strip())
                result_segments.append(TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    confidence=seg.avg_logprob if hasattr(seg, 'avg_logprob') else 0.9,
                ))
            
            return TranscriptionResult(
                text=" ".join(text_parts),
                language=info.language,
                duration_seconds=info.duration,
                segments=result_segments,
            )
            
        finally:
            if cleanup:
                Path(audio_path).unlink(missing_ok=True)

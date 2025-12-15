"""
Transcription Engine

Speech-to-text processing using faster-whisper.
"""

import asyncio
import logging
from typing import Any

from cara.config import Settings

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """
    Engine for speech-to-text transcription.
    
    Uses faster-whisper (CTranslate2) for efficient
    GPU-accelerated Whisper inference.
    """
    
    def __init__(self, stt_model: Any, settings: Settings):
        self.model = stt_model
        self.settings = settings
    
    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        include_timestamps: bool = True,
    ) -> dict:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (None for auto-detection)
            include_timestamps: Whether to include segment timestamps
            
        Returns:
            dict with keys: text, language, duration, segments
        """
        loop = asyncio.get_event_loop()
        
        # Run transcription in thread pool
        result = await loop.run_in_executor(
            None,
            lambda: self._transcribe_sync(audio_path, language, include_timestamps)
        )
        
        return result
    
    def _transcribe_sync(
        self,
        audio_path: str,
        language: str | None,
        include_timestamps: bool,
    ) -> dict:
        """Synchronous transcription logic."""
        
        # Transcribe with faster-whisper
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=self.settings.stt.beam_size,
            vad_filter=self.settings.stt.vad_filter,
            word_timestamps=include_timestamps,
        )
        
        # Collect segments
        segment_list = []
        full_text_parts = []
        
        for segment in segments:
            full_text_parts.append(segment.text)
            
            if include_timestamps:
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "confidence": segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0.9,
                })
        
        full_text = " ".join(full_text_parts).strip()
        
        return {
            "text": full_text,
            "language": info.language,
            "duration": info.duration,
            "segments": segment_list,
        }

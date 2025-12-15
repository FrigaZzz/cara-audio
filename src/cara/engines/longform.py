"""
Long-Form TTS Engine

Batch processing for long texts with checkpointing
and progress tracking.
"""

import asyncio
import logging
from typing import Any, AsyncIterator

from cara.config import Settings
from cara.utils.text import chunk_text_by_duration
from cara.utils.audio import numpy_to_pcm16

logger = logging.getLogger(__name__)


class LongFormTTSEngine:
    """
    Engine for long-form TTS generation.
    
    Processes long texts in chunks with:
    - Progress tracking
    - Checkpointing (resume capability)
    - Configurable chunk duration
    """
    
    def __init__(self, tts_model: Any, settings: Settings):
        self.tts = tts_model
        self.settings = settings
    
    async def generate(
        self,
        text: str,
        job_id: str,
        voice: str = "default",
        language: str = "it",
        speed: float = 1.0,
        start_chunk: int = 0,
    ) -> AsyncIterator[float]:
        """
        Generate TTS for long-form text.
        
        Args:
            text: Full text to convert
            job_id: Job identifier for tracking
            voice: Voice ID
            language: Language code
            speed: Speech speed
            start_chunk: Chunk index to resume from (for checkpointing)
            
        Yields:
            float: Progress (0.0 to 1.0)
        """
        # Split into duration-based chunks
        chunks = chunk_text_by_duration(
            text,
            target_duration=self.settings.performance.longform_chunk_duration,
        )
        
        total_chunks = len(chunks)
        logger.info(f"Job {job_id}: Processing {total_chunks} chunks")
        
        # Store generated audio chunks
        audio_parts = []
        
        for i, chunk in enumerate(chunks):
            if i < start_chunk:
                # Skip already processed chunks (resume)
                continue
            
            if not chunk.strip():
                continue
            
            logger.debug(f"Job {job_id}: Chunk {i+1}/{total_chunks}")
            
            try:
                # Generate audio for chunk
                audio = await self._generate_chunk(chunk, language, speed)
                audio_parts.append(audio)
                
                # TODO: Save checkpoint here for resume capability
                # await self._save_checkpoint(job_id, i, audio)
                
            except Exception as e:
                logger.error(f"Job {job_id}: Error on chunk {i}: {e}")
                raise
            
            # Yield progress
            progress = (i + 1) / total_chunks
            yield progress
        
        # TODO: Concatenate and save final audio
        # await self._save_final_audio(job_id, audio_parts)
        
        logger.info(f"Job {job_id}: Generation complete")
    
    async def _generate_chunk(
        self,
        text: str,
        language: str,
        speed: float,
    ) -> Any:
        """Generate audio for a single chunk."""
        loop = asyncio.get_event_loop()
        
        audio = await loop.run_in_executor(
            None,
            lambda: self.tts.tts(
                text=text,
                language=language,
                speed=speed,
            )
        )
        
        return audio
    
    async def _save_checkpoint(self, job_id: str, chunk_index: int, audio: Any) -> None:
        """Save checkpoint for resume capability."""
        # TODO: Implement with Redis or file storage
        pass
    
    async def _save_final_audio(self, job_id: str, audio_parts: list) -> None:
        """Concatenate and save final audio."""
        # TODO: Implement concatenation and storage
        pass

"""
Streaming TTS Engine

Real-time text-to-speech with chunk-by-chunk generation
for low-latency audio streaming.
"""

import asyncio
import logging
from typing import AsyncIterator, Any

from cara.config import Settings
from cara.utils.text import chunk_text_by_sentences
from cara.utils.audio import numpy_to_pcm16

logger = logging.getLogger(__name__)


class StreamingTTSEngine:
    """
    Engine for real-time TTS streaming.
    
    Generates audio incrementally by:
    1. Splitting text into sentence chunks
    2. Generating audio for each chunk
    3. Yielding audio bytes as they become available
    
    This approach provides low latency for the first audio chunk
    while maintaining natural speech prosody at sentence boundaries.
    """
    
    def __init__(self, tts_model: Any, settings: Settings):
        self.tts = tts_model
        self.settings = settings
    
    async def stream(
        self,
        text: str,
        voice: str = "default",
        language: str = "it",
        speed: float = 1.0,
    ) -> AsyncIterator[bytes]:
        """
        Stream TTS audio chunks.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID (for XTTS, this would be a speaker embedding)
            language: Language code
            speed: Speech speed multiplier
            
        Yields:
            bytes: PCM16 audio chunks
        """
        # Split text into sentence chunks for natural breaks
        chunks = chunk_text_by_sentences(text)
        
        logger.debug(f"Streaming {len(chunks)} text chunks")
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            logger.debug(f"Generating chunk {i+1}/{len(chunks)}: {chunk[:30]}...")
            
            # Generate audio for this chunk
            try:
                audio_data = await self._generate_chunk(chunk, language, speed)
                
                # Convert to PCM16 bytes
                pcm_bytes = numpy_to_pcm16(audio_data)
                
                # Yield in smaller sub-chunks for smoother streaming
                chunk_size = self.settings.streaming.chunk_size
                for j in range(0, len(pcm_bytes), chunk_size):
                    yield pcm_bytes[j:j + chunk_size]
                    
            except Exception as e:
                logger.error(f"Error generating chunk {i}: {e}")
                # Continue with next chunk instead of failing completely
                continue
    
    async def _generate_chunk(
        self,
        text: str,
        language: str,
        speed: float,
    ) -> Any:
        """Generate audio for a single text chunk."""
        loop = asyncio.get_event_loop()
        
        # Run TTS in thread pool to not block event loop
        # ChatterboxEngine uses generate() method
        audio = await loop.run_in_executor(
            None,
            lambda: self.tts.generate(
                text=text,
                language=language,
            ).squeeze().cpu().numpy()
        )
        
        return audio
    
    async def generate_full(
        self,
        text: str,
        voice: str = "default",
        language: str = "it",
        speed: float = 1.0,
    ) -> bytes:
        """
        Generate complete audio for the entire text.
        
        This is a convenience method that collects all chunks
        and returns the full audio.
        """
        audio_chunks = []
        async for chunk in self.stream(text, voice, language, speed):
            audio_chunks.append(chunk)
        
        return b"".join(audio_chunks)

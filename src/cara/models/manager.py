"""
Model Manager - Centralized model loading and lifecycle management

This module provides a clean interface for managing TTS and STT models,
making it easy to swap models and configure parameters.
"""

import asyncio
import logging
from typing import Any

from cara.config import get_settings

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Centralized manager for TTS and STT models.
    
    Handles:
    - Model loading with GPU pinning
    - Model lifecycle (load/unload)
    - Thread-safe access to models
    """
    
    def __init__(self):
        self._tts_model: Any | None = None
        self._stt_model: Any | None = None
        self._lock = asyncio.Lock()
        self._loaded = False
    
    @property
    def tts(self) -> Any:
        """Get the TTS model instance"""
        if self._tts_model is None:
            raise RuntimeError("TTS model not loaded. Call load_all() first.")
        return self._tts_model
    
    @property
    def stt(self) -> Any:
        """Get the STT model instance"""
        if self._stt_model is None:
            raise RuntimeError("STT model not loaded. Call load_all() first.")
        return self._stt_model
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return self._loaded
    
    async def load_all(self) -> None:
        """Load all models concurrently"""
        async with self._lock:
            if self._loaded:
                logger.info("Models already loaded, skipping...")
                return
            
            # Load models concurrently
            await asyncio.gather(
                self._load_tts(),
                self._load_stt(),
            )
            self._loaded = True
    
    async def _load_tts(self) -> None:
        """Load TTS model"""
        settings = get_settings()
        logger.info(f"Loading Chatterbox TTS ({settings.tts.model_variant})...")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self._tts_model = await loop.run_in_executor(
            None,
            self._load_tts_sync,
        )
        logger.info("✅ TTS model loaded")
    
    def _load_tts_sync(self) -> Any:
        """Synchronous TTS model loading"""
        from cara.engines.chatterbox_engine import ChatterboxEngine
        
        settings = get_settings()
        
        return ChatterboxEngine(
            variant=settings.tts.model_variant,
            device=settings.tts.device,
        )
    
    async def _load_stt(self) -> None:
        """Load STT model"""
        settings = get_settings()
        logger.info(f"Loading STT model: whisper-{settings.stt.model_size}")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self._stt_model = await loop.run_in_executor(
            None,
            self._load_stt_sync,
        )
        logger.info("✅ STT model loaded")
    
    def _load_stt_sync(self) -> Any:
        """Synchronous STT model loading"""
        from faster_whisper import WhisperModel
        
        settings = get_settings()
        
        # Initialize Whisper model
        model = WhisperModel(
            settings.stt.model_size,
            device=settings.stt.device,
            compute_type=settings.stt.compute_type,
        )
        
        return model
    
    async def unload_all(self) -> None:
        """Unload all models and free GPU memory"""
        async with self._lock:
            if not self._loaded:
                return
            
            logger.info("Unloading models...")
            
            # Clear references
            self._tts_model = None
            self._stt_model = None
            
            # Force garbage collection to free GPU memory
            import gc
            import torch
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._loaded = False
            logger.info("✅ Models unloaded")
    
    async def reload_tts(self, model_variant: str | None = None) -> None:
        """
        Hot-reload TTS model (useful for model swapping at runtime).
        
        Args:
            model_variant: Optional new model variant ('multilingual', 'turbo', 'original').
                          If None, reloads current model.
        """
        async with self._lock:
            if model_variant:
                # Update settings via environment variable
                import os
                os.environ["TTS_MODEL_VARIANT"] = model_variant
                # Clear cached settings
                from cara.config import get_settings
                get_settings.cache_clear()
            
            logger.info(f"Hot-reloading TTS model...")
            self._tts_model = None
            
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            await self._load_tts()
    
    async def reload_stt(self, model_size: str | None = None) -> None:
        """
        Hot-reload STT model (useful for model swapping at runtime).
        
        Args:
            model_size: Optional new model size. If None, reloads current model.
        """
        async with self._lock:
            if model_size:
                import os
                os.environ["STT_MODEL_SIZE"] = model_size
                from cara.config import get_settings
                get_settings.cache_clear()
            
            logger.info(f"Hot-reloading STT model...")
            self._stt_model = None
            
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            await self._load_stt()

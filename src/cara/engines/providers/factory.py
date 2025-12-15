"""
Provider Factory.

Creates TTS and STT providers based on configuration.
"""

import logging
from typing import Literal

from .tts_base import TTSProvider
from .stt_base import STTProvider

logger = logging.getLogger(__name__)


# Available providers
TTS_PROVIDERS = {
    "chatterbox": "cara_audio.engines.providers.chatterbox_tts.ChatterboxTTSProvider",
    # Future providers:
    # "elevenlabs": "cara_audio.engines.providers.elevenlabs_tts.ElevenLabsTTSProvider",
    # "azure": "cara_audio.engines.providers.azure_tts.AzureTTSProvider",
    # "google": "cara_audio.engines.providers.google_tts.GoogleTTSProvider",
}

STT_PROVIDERS = {
    "whisper": "cara_audio.engines.providers.whisper_stt.WhisperSTTProvider",
    # Future providers:
    # "deepgram": "cara_audio.engines.providers.deepgram_stt.DeepgramSTTProvider",
    # "azure": "cara_audio.engines.providers.azure_stt.AzureSTTProvider",
    # "google": "cara_audio.engines.providers.google_stt.GoogleSTTProvider",
}


def _import_class(class_path: str):
    """Dynamically import a class from a module path."""
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_tts_provider(
    provider: str = "chatterbox",
    **kwargs,
) -> TTSProvider:
    """
    Create a TTS provider instance.
    
    Args:
        provider: Provider name (chatterbox, elevenlabs, azure, google)
        **kwargs: Provider-specific configuration
        
    Returns:
        TTSProvider instance
    """
    if provider not in TTS_PROVIDERS:
        raise ValueError(f"Unknown TTS provider: {provider}. Available: {list(TTS_PROVIDERS.keys())}")
    
    provider_class = _import_class(TTS_PROVIDERS[provider])
    logger.info(f"Creating TTS provider: {provider}")
    
    return provider_class(**kwargs)


def create_stt_provider(
    provider: str = "whisper",
    **kwargs,
) -> STTProvider:
    """
    Create an STT provider instance.
    
    Args:
        provider: Provider name (whisper, deepgram, azure, google)
        **kwargs: Provider-specific configuration
        
    Returns:
        STTProvider instance
    """
    if provider not in STT_PROVIDERS:
        raise ValueError(f"Unknown STT provider: {provider}. Available: {list(STT_PROVIDERS.keys())}")
    
    provider_class = _import_class(STT_PROVIDERS[provider])
    logger.info(f"Creating STT provider: {provider}")
    
    return provider_class(**kwargs)


def list_tts_providers() -> list[str]:
    """List available TTS providers."""
    return list(TTS_PROVIDERS.keys())


def list_stt_providers() -> list[str]:
    """List available STT providers."""
    return list(STT_PROVIDERS.keys())

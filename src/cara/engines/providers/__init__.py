"""
Provider interfaces and implementations.
"""

from .tts_base import TTSProvider, TTSResult
from .stt_base import STTProvider, TranscriptionResult, TranscriptionSegment
from .factory import create_tts_provider, create_stt_provider, list_tts_providers, list_stt_providers

# Concrete implementations
from .chatterbox_tts import ChatterboxTTSProvider
from .whisper_stt import WhisperSTTProvider

__all__ = [
    # Base classes
    "TTSProvider",
    "TTSResult",
    "STTProvider",
    "TranscriptionResult",
    "TranscriptionSegment",
    # Factory
    "create_tts_provider",
    "create_stt_provider",
    "list_tts_providers",
    "list_stt_providers",
    # Implementations
    "ChatterboxTTSProvider",
    "WhisperSTTProvider",
]

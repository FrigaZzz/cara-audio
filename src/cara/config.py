"""
CARA Audio Backend - Configuration Module

All settings are configurable via environment variables.
Designed for easy model swapping and parameter tuning.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TTSConfig(BaseSettings):
    """TTS Model Configuration - Chatterbox (MIT licensed)"""
    
    model_config = SettingsConfigDict(env_prefix="TTS_")
    
    # Provider selection
    provider: Literal["chatterbox", "elevenlabs", "azure", "google"] = Field(
        default="chatterbox",
        description="TTS provider (chatterbox=local, others=cloud)"
    )
    
    # Model variant: multilingual (Italian), turbo (English, faster), original (English)
    model_variant: Literal["multilingual", "turbo", "original"] = Field(
        default="multilingual",
        description="Chatterbox model variant. Use 'multilingual' for Italian support."
    )
    
    # Hardware settings
    device: Literal["cuda", "cpu"] = Field(
        default="cuda",
        description="Device for inference"
    )
    
    # Voice settings
    default_language: str = Field(
        default="it",
        description="Default language for TTS (ISO 639-1)"
    )
    speaker_reference_path: str | None = Field(
        default=None,
        description="Path to speaker reference WAV for voice cloning (10-15s recommended)"
    )


class STTConfig(BaseSettings):
    """STT Model Configuration - swap Whisper models easily!"""
    
    model_config = SettingsConfigDict(env_prefix="STT_")
    
    # Provider selection
    provider: Literal["whisper", "deepgram", "azure", "google"] = Field(
        default="whisper",
        description="STT provider (whisper=local, others=cloud)"
    )
    
    # Model selection - change this to swap STT models
    model_size: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = Field(
        default="large-v3",
        description="Whisper model size. Larger = better quality, more VRAM"
    )
    
    # Hardware settings
    device: Literal["cuda", "cpu"] = Field(
        default="cuda",
        description="Device for inference"
    )
    compute_type: Literal["float16", "float32", "int8", "int8_float16"] = Field(
        default="float16",
        description="Compute precision"
    )
    
    # Transcription settings
    default_language: str = Field(
        default="it",
        description="Default language for transcription (ISO 639-1)"
    )
    beam_size: int = Field(
        default=5,
        description="Beam size for decoding"
    )
    vad_filter: bool = Field(
        default=True,
        description="Enable Voice Activity Detection filter"
    )


class AudioConfig(BaseSettings):
    """Audio processing configuration"""
    
    model_config = SettingsConfigDict(env_prefix="AUDIO_")
    
    sample_rate: int = Field(
        default=24000,
        description="Audio sample rate in Hz"
    )
    channels: int = Field(
        default=1,
        description="Number of audio channels (1=mono, 2=stereo)"
    )
    format: Literal["pcm16", "pcm32", "float32"] = Field(
        default="pcm16",
        description="Audio format"
    )


class StreamingConfig(BaseSettings):
    """Streaming configuration"""
    
    model_config = SettingsConfigDict(env_prefix="STREAMING_")
    
    chunk_size: int = Field(
        default=4096,
        description="Chunk size in bytes for streaming"
    )
    buffer_seconds: float = Field(
        default=0.5,
        description="Buffer duration in seconds"
    )


class RedisConfig(BaseSettings):
    """Redis job queue configuration"""
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )


class JobConfig(BaseSettings):
    """Job queue configuration"""
    
    model_config = SettingsConfigDict(env_prefix="JOB_")
    
    ttl_hours: int = Field(
        default=24,
        description="Job time-to-live in hours"
    )


class ServerConfig(BaseSettings):
    """Server configuration"""
    
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    
    # Security
    api_key: str | None = Field(default=None, description="API key for authentication")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="Comma-separated list of allowed origins"
    )
    
    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]


class PerformanceConfig(BaseSettings):
    """Performance tuning configuration"""
    
    max_streaming_sessions: int = Field(
        default=5,
        description="Max concurrent streaming sessions per GPU"
    )
    longform_chunk_duration: int = Field(
        default=20,
        description="Duration of each chunk in long-form TTS (seconds)"
    )
    max_concurrent_jobs: int = Field(
        default=2,
        description="Max concurrent long-form jobs"
    )
    preload_models: bool = Field(
        default=True,
        description="Preload models on startup"
    )


class Settings(BaseSettings):
    """Main settings class that aggregates all configuration"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Sub-configurations
    tts: TTSConfig = Field(default_factory=TTSConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    job: JobConfig = Field(default_factory=JobConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

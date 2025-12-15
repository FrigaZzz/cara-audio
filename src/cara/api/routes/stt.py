"""
STT (Speech-to-Text) API endpoints.

Provides transcription of audio to text using faster-whisper.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field

from cara.config import get_settings
from cara.models import model_manager
from cara.engines.transcription import TranscriptionEngine

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Response Models
# =============================================================================

class TranscriptionSegment(BaseModel):
    """A segment of transcribed text with timing"""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Confidence score (0-1)")


class TranscriptionResponse(BaseModel):
    """Full transcription response"""
    text: str = Field(..., description="Full transcribed text")
    language: str = Field(..., description="Detected or specified language")
    duration: float = Field(..., description="Audio duration in seconds")
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    
    model_config = {"json_schema_extra": {"example": {
        "text": "Ciao, come stai oggi?",
        "language": "it",
        "duration": 2.5,
        "segments": [
            {"start": 0.0, "end": 1.2, "text": "Ciao,", "confidence": 0.95},
            {"start": 1.2, "end": 2.5, "text": "come stai oggi?", "confidence": 0.92}
        ]
    }}}


# =============================================================================
# REST Endpoints
# =============================================================================

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: Annotated[UploadFile, File(description="Audio file to transcribe")],
    language: Annotated[str | None, Form(description="Language code (auto-detect if not provided)")] = None,
    timestamps: Annotated[bool, Form(description="Include word-level timestamps")] = True,
):
    """
    Transcribe audio file to text.
    
    Supports: WAV, MP3, M4A, WebM, OGG, FLAC
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    settings = get_settings()
    
    # Validate file type
    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/webm", 
                     "audio/ogg", "audio/flac", "audio/x-m4a", "audio/mp4"]
    
    if audio.content_type and audio.content_type not in allowed_types:
        logger.warning(f"Unexpected content type: {audio.content_type}, attempting anyway")
    
    try:
        # Save uploaded file to temp location
        suffix = Path(audio.filename or "audio.wav").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Initialize transcription engine
        engine = TranscriptionEngine(model_manager.stt, settings)
        
        # Run transcription
        result = await engine.transcribe(
            tmp_path,
            language=language or settings.stt.default_language,
            include_timestamps=timestamps,
        )
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        
        return TranscriptionResponse(
            text=result["text"],
            language=result["language"],
            duration=result["duration"],
            segments=[
                TranscriptionSegment(**seg) for seg in result.get("segments", [])
            ],
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload")
async def reload_model(
    model_size: Annotated[str | None, Query(
        description="New model size (tiny, base, small, medium, large-v2, large-v3)"
    )] = None
):
    """
    Hot-reload the STT model.
    
    Useful for swapping model sizes at runtime.
    """
    try:
        await model_manager.reload_stt(model_size)
        settings = get_settings()
        return {
            "success": True,
            "model": f"whisper-{settings.stt.model_size}",
        }
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/languages")
async def list_languages():
    """List supported languages for transcription."""
    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "it", "name": "Italian"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "nl", "name": "Dutch"},
            {"code": "pl", "name": "Polish"},
            {"code": "ru", "name": "Russian"},
            {"code": "ja", "name": "Japanese"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ko", "name": "Korean"},
        ],
        "note": "Whisper supports 99+ languages. Set language=None for auto-detection."
    }

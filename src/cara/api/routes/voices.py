"""
Voice Management API routes.

Handles voice profile CRUD operations:
- Upload voice samples for cloning
- List available voices
- Delete voice profiles
"""

import asyncio
import io
import json
import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class VoiceMetadata(BaseModel):
    """Voice profile metadata"""
    id: str
    name: str
    description: str = ""
    language: str = "multilingual"
    created_at: str | None = None
    sample_duration_seconds: float | None = None


class VoiceUploadResponse(BaseModel):
    """Response after uploading a voice sample"""
    success: bool
    voice: VoiceMetadata
    message: str


class VoiceListResponse(BaseModel):
    """Response for listing voices"""
    voices: list[VoiceMetadata]
    total: int


# =============================================================================
# Helper Functions
# =============================================================================

def get_voices_dir() -> Path:
    """Get the voices directory path."""
    base_path = Path(__file__).parent.parent.parent.parent.parent
    return base_path / "assets" / "voices"


def get_voice_path(voice_id: str) -> Path:
    """Get path to a specific voice profile directory."""
    return get_voices_dir() / voice_id


def load_voice_metadata(voice_id: str) -> VoiceMetadata | None:
    """Load metadata for a voice profile."""
    voice_dir = get_voice_path(voice_id)
    metadata_file = voice_dir / "metadata.json"
    sample_file = voice_dir / "sample.wav"
    
    if not sample_file.exists():
        return None
    
    if metadata_file.exists():
        try:
            data = json.loads(metadata_file.read_text())
            return VoiceMetadata(id=voice_id, **data)
        except Exception as e:
            logger.warning(f"Failed to load metadata for {voice_id}: {e}")
    
    return VoiceMetadata(id=voice_id, name=voice_id)


# =============================================================================
# Voice Management Endpoints
# =============================================================================

@router.post("/upload", response_model=VoiceUploadResponse)
async def upload_voice(
    file: Annotated[UploadFile, File(description="Voice sample WAV file (10-15 seconds recommended)")],
    name: Annotated[str, Form(description="Display name for the voice")],
    description: Annotated[str, Form(description="Optional description")] = "",
    language: Annotated[str, Form(description="Primary language of the voice")] = "multilingual",
):
    """
    Upload a voice sample for cloning.
    
    The audio file should be:
    - WAV format (16-bit, mono preferred)
    - 10-15 seconds of clear speech
    - Minimal background noise
    - Single speaker only
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a WAV, MP3, or FLAC file."
        )
    
    # Generate unique voice ID
    voice_id = str(uuid.uuid4())[:8]
    voice_dir = get_voice_path(voice_id)
    
    try:
        # Create voice directory
        voice_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the audio file
        sample_path = voice_dir / "sample.wav"
        content = await file.read()
        
        # If not WAV, we should convert (for now, just save as-is)
        # TODO: Add audio conversion using pydub/librosa
        sample_path.write_bytes(content)
        
        # Calculate approximate duration
        import wave
        try:
            with wave.open(str(sample_path), 'rb') as wf:
                duration = wf.getnframes() / wf.getframerate()
        except:
            duration = None
        
        # Save metadata
        from datetime import datetime
        metadata = {
            "name": name,
            "description": description,
            "language": language,
            "created_at": datetime.now().isoformat(),
            "sample_duration_seconds": duration,
            "original_filename": file.filename,
        }
        
        metadata_path = voice_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        
        logger.info(f"Voice profile created: {voice_id} ({name})")
        
        return VoiceUploadResponse(
            success=True,
            voice=VoiceMetadata(id=voice_id, **metadata),
            message=f"Voice profile '{name}' created successfully. Use voice_id='{voice_id}' in TTS requests."
        )
        
    except Exception as e:
        # Cleanup on failure
        if voice_dir.exists():
            import shutil
            shutil.rmtree(voice_dir, ignore_errors=True)
        
        logger.error(f"Voice upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=VoiceListResponse)
async def list_all_voices():
    """List all available voice profiles."""
    voices_dir = get_voices_dir()
    voices = []
    
    # Check for default voice
    default_speaker = voices_dir / "default_speaker.wav"
    if default_speaker.exists():
        voices.append(VoiceMetadata(
            id="default",
            name="Default Voice",
            description="Built-in default voice",
            language="multilingual",
        ))
    
    # Scan for custom voices
    if voices_dir.exists():
        for item in voices_dir.iterdir():
            if item.is_dir():
                voice = load_voice_metadata(item.name)
                if voice:
                    voices.append(voice)
    
    return VoiceListResponse(voices=voices, total=len(voices))


@router.get("/{voice_id}", response_model=VoiceMetadata)
async def get_voice(voice_id: str):
    """Get details of a specific voice profile."""
    if voice_id == "default":
        default_speaker = get_voices_dir() / "default_speaker.wav"
        if default_speaker.exists():
            return VoiceMetadata(
                id="default",
                name="Default Voice",
                description="Built-in default voice",
                language="multilingual",
            )
        raise HTTPException(status_code=404, detail="Default voice not found")
    
    voice = load_voice_metadata(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice profile '{voice_id}' not found")
    
    return voice


@router.get("/{voice_id}/sample")
async def get_voice_sample(voice_id: str):
    """Download the voice sample audio file."""
    if voice_id == "default":
        sample_path = get_voices_dir() / "default_speaker.wav"
    else:
        sample_path = get_voice_path(voice_id) / "sample.wav"
    
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Voice sample not found")
    
    return FileResponse(
        sample_path,
        media_type="audio/wav",
        filename=f"{voice_id}_sample.wav"
    )


@router.delete("/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a voice profile."""
    if voice_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete the default voice")
    
    voice_dir = get_voice_path(voice_id)
    
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail=f"Voice profile '{voice_id}' not found")
    
    try:
        import shutil
        shutil.rmtree(voice_dir)
        logger.info(f"Voice profile deleted: {voice_id}")
        
        return {"success": True, "message": f"Voice profile '{voice_id}' deleted"}
        
    except Exception as e:
        logger.error(f"Failed to delete voice {voice_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{voice_id}/preview")
async def preview_voice(
    voice_id: str,
    text: Annotated[str, Form(description="Text to synthesize")] = "Hello, this is a voice preview test.",
    language: Annotated[str, Form(description="Language code")] = "en",
):
    """
    Generate a quick preview using a voice profile.
    
    This is a convenience endpoint for testing voices before using them.
    """
    from cara.models import model_manager
    from cara.utils.audio import audio_to_wav_bytes
    
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Get voice sample path
    if voice_id == "default":
        sample_path = get_voices_dir() / "default_speaker.wav"
    else:
        sample_path = get_voice_path(voice_id) / "sample.wav"
    
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Voice sample not found")
    
    try:
        engine = model_manager.tts
        loop = asyncio.get_event_loop()
        
        def generate():
            wav_tensor = engine.generate(
                text=text,
                language=language,
                speaker_wav=str(sample_path),
                exaggeration=0.5,
                temperature=0.8,
                cfg_weight=0.5,
            )
            return wav_tensor.squeeze().cpu().numpy()
        
        audio_np = await loop.run_in_executor(None, generate)
        wav_bytes = audio_to_wav_bytes(audio_np.tolist(), sample_rate=engine.sample_rate)
        
        from fastapi.responses import Response
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename={voice_id}_preview.wav"}
        )
        
    except Exception as e:
        logger.error(f"Voice preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

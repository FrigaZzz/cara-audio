"""
TTS (Text-to-Speech) API endpoints.

Includes:
- REST endpoint for simple TTS
- WebSocket endpoint for real-time streaming
"""

import asyncio
import io
import logging
from typing import Annotated

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cara.config import get_settings
from cara.models import model_manager
from cara.engines.streaming import StreamingTTSEngine
from cara.utils.audio import audio_to_wav_bytes

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class TTSRequest(BaseModel):
    """TTS request body with advanced generation parameters."""
    
    # Core parameters
    text: str = Field(..., description="Text to convert to speech", min_length=1, max_length=10000)
    language: str = Field(default="it", description="Language code (ISO 639-1)")
    voice_id: str | None = Field(default=None, description="Voice profile ID for cloning")
    
    # Emotion mode - switches to Turbo model (English only)
    use_emotion_model: bool = Field(
        default=False,
        description="Use Turbo model for emotion tags ([laugh], [sigh], etc.). Auto-detected if tags found in text. English only."
    )
    
    # Generation parameters (Chatterbox-specific)
    exaggeration: float = Field(
        default=0.5, ge=0.25, le=2.0,
        description="Emotion intensity: 0.3-0.4 (neutral), 0.5 (balanced), 0.7+ (expressive), 1.0+ (dramatic)"
    )
    temperature: float = Field(
        default=0.8, ge=0.05, le=5.0,
        description="Creativity/randomness: 0.4-0.6 (consistent), 0.8 (balanced), 1.0+ (creative)"
    )
    cfg_weight: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Voice fidelity: higher = more faithful to reference voice"
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducible generations"
    )
    
    # Audio format options
    format: str = Field(default="wav", pattern="^(wav|mp3)$", description="Output audio format")
    sample_rate: int = Field(default=24000, description="Output sample rate in Hz")
    
    model_config = {"json_schema_extra": {"example": {
        "text": "Hello there! [laugh] How are you today?",
        "language": "en",
        "use_emotion_model": True,
        "exaggeration": 0.6,
        "temperature": 0.8,
        "cfg_weight": 0.5,
        "seed": None
    }}}


class TTSResponse(BaseModel):
    """TTS response metadata"""
    success: bool
    duration_seconds: float
    sample_rate: int
    format: str
    generation_params: dict | None = None


# =============================================================================
# REST Endpoints
# =============================================================================

@router.post("/generate")
async def generate_tts(request: TTSRequest):
    """
    Generate TTS audio from text.
    
    Supports dynamic model switching:
    - Default: Multilingual model (Italian + 22 languages)
    - Emotion mode: Turbo model (English only, supports [laugh], [sigh], etc.)
    
    Returns WAV audio as a streaming response.
    """
    import re
    
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    settings = get_settings()
    
    # Detect emotion tags in text
    emotion_pattern = r'\[(laugh|chuckle|sigh|gasp|cough|clearing throat|sniff|groan)\]'
    has_emotion_tags = bool(re.search(emotion_pattern, request.text, re.IGNORECASE))
    
    # Determine if we need emotion model
    needs_emotion_model = request.use_emotion_model or has_emotion_tags
    
    # If emotion mode needed but language isn't English, warn and continue
    effective_language = request.language
    model_switched = False
    
    if needs_emotion_model:
        if request.language != "en":
            logger.warning(f"Emotion tags detected but language is '{request.language}'. Switching to English for Turbo model.")
            effective_language = "en"
        
        # Check if we need to switch models
        current_variant = settings.tts.model_variant
        if current_variant != "turbo":
            logger.info(f"🔄 Switching from {current_variant} to turbo for emotion tags...")
            await model_manager.reload_tts("turbo")
            model_switched = True
    else:
        # Ensure we're on multilingual for non-English
        current_variant = settings.tts.model_variant
        if current_variant == "turbo" and request.language != "en":
            logger.info(f"🔄 Switching from turbo to multilingual for {request.language}...")
            await model_manager.reload_tts("multilingual")
            model_switched = True
    
    try:
        # Get Chatterbox engine (may have been reloaded)
        engine = model_manager.tts
        
        # Run TTS in thread pool to not block
        loop = asyncio.get_event_loop()
        
        def generate_audio():
            """Generate audio using Chatterbox with advanced parameters"""
            from pathlib import Path
            import numpy as np
            
            # Determine speaker wav file
            base_path = Path(__file__).parent.parent.parent.parent.parent
            print(request.voice_id)
            if request.voice_id:
                # Use custom voice profile
                voice_path = base_path / "assets" / "voices" / request.voice_id / "sample.wav"
                speaker_path = str(voice_path) if voice_path.exists() else None
            else:
                # Use default speaker
                default_speaker = base_path / "assets" / "voices" / "default_speaker.wav"
                speaker_path = str(default_speaker) if default_speaker.exists() else None
            
            # Generate audio with Chatterbox (with advanced parameters)
            wav_tensor = engine.generate(
                text=request.text,
                language=effective_language,
                speaker_wav=speaker_path,
                exaggeration=request.exaggeration,
                temperature=request.temperature,
                cfg_weight=request.cfg_weight,
                seed=request.seed,
            )
            
            # Convert tensor to numpy array
            audio_np = wav_tensor.squeeze().cpu().numpy()
            
            # Normalize to float32 range [-1, 1] if needed
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)
            
            return audio_np.tolist(), engine.sample_rate
        
        wav_data, sample_rate = await loop.run_in_executor(None, generate_audio)
        
        # Convert to WAV bytes
        wav_bytes = audio_to_wav_bytes(
            wav_data,
            sample_rate=sample_rate,
        )
        
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Audio-Duration": str(len(wav_data) / sample_rate),
                "X-Model-Used": "turbo" if needs_emotion_model else settings.tts.model_variant,
                "X-Model-Switched": str(model_switched).lower(),
            }
        )
        
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WebSocket Streaming Endpoint
# =============================================================================

@router.websocket("/stream")
async def stream_tts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time TTS streaming.
    
    Protocol:
    1. Client connects
    2. Client sends JSON: {"text": "...", "voice": "...", "language": "...", "speed": 1.0}
    3. Server streams binary audio chunks
    4. Server sends JSON: {"done": true, "duration": ...} when complete
    
    Client can send "interrupt" message to stop current generation.
    """
    await websocket.accept()
    
    if not model_manager.is_loaded:
        await websocket.send_json({"error": "Models not loaded"})
        await websocket.close(code=1013)
        return
    
    settings = get_settings()
    engine = StreamingTTSEngine(model_manager.tts, settings)
    
    try:
        while True:
            # Wait for TTS request
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            text = data.get("text", "")
            if not text:
                await websocket.send_json({"error": "No text provided"})
                continue
            
            voice = data.get("voice", "default")
            language = data.get("language", settings.tts.default_language)
            speed = data.get("speed", 1.0)
            
            logger.info(f"Streaming TTS: {text[:50]}...")
            
            # Stream audio chunks
            total_duration = 0.0
            async for chunk in engine.stream(text, voice=voice, language=language, speed=speed):
                await websocket.send_bytes(chunk)
                total_duration += len(chunk) / (settings.audio.sample_rate * 2)  # 16-bit = 2 bytes
            
            # Send completion message
            await websocket.send_json({
                "done": True,
                "duration": total_duration,
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


# =============================================================================
# Configuration Endpoints
# =============================================================================

@router.get("/voices")
async def list_voices():
    """List available voices for TTS."""
    from pathlib import Path
    import json
    
    base_path = Path(__file__).parent.parent.parent.parent.parent
    voices_dir = base_path / "assets" / "voices"
    
    voices = []
    
    # Add default voice
    default_speaker = voices_dir / "default_speaker.wav"
    if default_speaker.exists():
        voices.append({
            "id": "default",
            "name": "Default Voice",
            "language": "multilingual",
            "is_default": True,
        })
    
    # Scan for custom voice profiles
    if voices_dir.exists():
        for voice_dir in voices_dir.iterdir():
            if voice_dir.is_dir():
                metadata_file = voice_dir / "metadata.json"
                sample_file = voice_dir / "sample.wav"
                
                if sample_file.exists():
                    metadata = {"name": voice_dir.name, "language": "multilingual"}
                    if metadata_file.exists():
                        try:
                            metadata = json.loads(metadata_file.read_text())
                        except:
                            pass
                    
                    voices.append({
                        "id": voice_dir.name,
                        "name": metadata.get("name", voice_dir.name),
                        "language": metadata.get("language", "multilingual"),
                        "description": metadata.get("description", ""),
                        "is_default": False,
                    })
    
    return {
        "voices": voices,
        "total": len(voices),
    }


@router.get("/emotion-tags")
async def get_emotion_tags():
    """
    Get available emotion/paralinguistic tags for text.
    
    ⚠️ IMPORTANT: Emotion tags only work with 'turbo' or 'original' models (English only).
    For Italian (multilingual model), use the 'exaggeration' parameter instead.
    
    Example: "Hello there [laugh], how are you today?"
    """
    settings = get_settings()
    
    return {
        "current_model": settings.tts.model_variant,
        "tags_supported": settings.tts.model_variant in ("turbo", "original"),
        "tags": [
            {"tag": "[laugh]", "description": "Laughter", "intensity": "medium"},
            {"tag": "[chuckle]", "description": "Light chuckle", "intensity": "low"},
            {"tag": "[sigh]", "description": "Sigh/exhale", "intensity": "medium"},
            {"tag": "[gasp]", "description": "Surprised gasp", "intensity": "high"},
            {"tag": "[cough]", "description": "Cough", "intensity": "low"},
            {"tag": "[clearing throat]", "description": "Throat clearing", "intensity": "low"},
            {"tag": "[sniff]", "description": "Sniffing", "intensity": "low"},
            {"tag": "[groan]", "description": "Groan/moan", "intensity": "medium"},
        ],
        "usage": "Insert tags directly in your text, e.g., 'Hello [laugh], how are you?'",
        "warning": "Emotion tags ONLY work with 'turbo' or 'original' models (English only). For Italian/multilingual, use the 'exaggeration' parameter (0.6-1.0 for expressive speech).",
        "alternative_for_italian": {
            "parameter": "exaggeration",
            "neutral": 0.35,
            "warm": 0.5,
            "expressive": 0.7,
            "dramatic": 1.0,
        }
    }


@router.get("/parameters")
async def get_generation_parameters():
    """
    Get documentation for TTS generation parameters.
    
    Use these to fine-tune voice output for your use case.
    """
    return {
        "parameters": {
            "exaggeration": {
                "range": [0.25, 2.0],
                "default": 0.5,
                "description": "Controls emotion intensity in the voice",
                "presets": {
                    "monotone": 0.25,
                    "neutral": 0.35,
                    "balanced": 0.5,
                    "expressive": 0.7,
                    "dramatic": 1.0,
                    "over_the_top": 1.5,
                }
            },
            "temperature": {
                "range": [0.05, 5.0],
                "default": 0.8,
                "description": "Controls creativity and randomness of speech patterns",
                "presets": {
                    "consistent": 0.4,
                    "balanced": 0.8,
                    "creative": 1.2,
                    "highly_variable": 2.0,
                }
            },
            "cfg_weight": {
                "range": [0.0, 1.0],
                "default": 0.5,
                "description": "Voice fidelity to reference audio (higher = more faithful)",
                "presets": {
                    "loose": 0.2,
                    "balanced": 0.5,
                    "faithful": 0.8,
                    "exact": 1.0,
                }
            },
            "seed": {
                "type": "integer",
                "default": None,
                "description": "Random seed for reproducible generations"
            }
        },
        "cara_recommended": {
            "warm_caring": {"exaggeration": 0.6, "temperature": 0.7, "cfg_weight": 0.6},
            "calm_professional": {"exaggeration": 0.4, "temperature": 0.6, "cfg_weight": 0.7},
            "cheerful_encouraging": {"exaggeration": 0.8, "temperature": 0.9, "cfg_weight": 0.5},
        }
    }


@router.post("/reload")
async def reload_model(
    model_variant: Annotated[str | None, Query(description="Model variant: 'multilingual', 'turbo', or 'original'")] = None
):
    """
    Hot-reload the TTS model.
    
    Useful for swapping models at runtime without restart.
    """
    try:
        await model_manager.reload_tts(model_variant)
        settings = get_settings()
        return {
            "success": True,
            "model": f"Chatterbox ({settings.tts.model_variant})",
        }
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

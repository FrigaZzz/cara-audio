"""
Health check endpoints for monitoring and observability.
"""

import torch
from fastapi import APIRouter

from cara.config import get_settings
from cara.models import model_manager

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and model availability.
    """
    settings = get_settings()
    
    return {
        "status": "healthy",
        "service": "cara-audio-backend",
        "version": "0.1.0",
        "models": {
            "tts": {
                "loaded": model_manager.is_loaded,
                "model": f"Chatterbox ({settings.tts.model_variant})",
                "device": settings.tts.device,
            },
            "stt": {
                "loaded": model_manager.is_loaded,
                "model": f"whisper-{settings.stt.model_size}",
                "device": settings.stt.device,
            },
        },
        "gpu": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness probe for Kubernetes.
    
    Returns 200 only if models are loaded and ready.
    """
    if not model_manager.is_loaded:
        return {"status": "not_ready", "reason": "models_not_loaded"}, 503
    
    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check():
    """Liveness probe for Kubernetes."""
    return {"status": "alive"}

"""
CARA Audio Backend - FastAPI Application

GPU-accelerated TTS + STT service with:
- Real-time streaming TTS via WebSocket
- Long-form TTS via job queue
- Speech-to-text transcription
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from cara.config import get_settings
from cara.api.routes import health, tts, stt, jobs, voices
from cara.models import model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.
    
    Handles model preloading on startup and cleanup on shutdown.
    """
    settings = get_settings()
    
    # Startup
    logger.info("🚀 Starting CARA Audio Backend...")
    logger.info(f"   TTS Model: Chatterbox ({settings.tts.model_variant})")
    logger.info(f"   STT Model: whisper-{settings.stt.model_size}")
    logger.info(f"   Device: {settings.tts.device}")
    
    if settings.performance.preload_models:
        logger.info("📦 Preloading models...")
        await model_manager.load_all()
        logger.info("✅ Models loaded and ready!")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down CARA Audio Backend...")
    await model_manager.unload_all()
    logger.info("✅ Cleanup complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="CARA Audio Backend",
        description="GPU-accelerated TTS + STT service for CARA",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.server.debug,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount Prometheus metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    # Register routes
    app.include_router(health.router, tags=["Health"])
    app.include_router(tts.router, prefix="/tts", tags=["TTS"])
    app.include_router(voices.router, prefix="/voices", tags=["Voices"])
    app.include_router(stt.router, prefix="/stt", tags=["STT"])
    app.include_router(jobs.router, prefix="/tts/jobs", tags=["Jobs"])
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "cara.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug,
        log_level=settings.server.log_level.lower(),
    )

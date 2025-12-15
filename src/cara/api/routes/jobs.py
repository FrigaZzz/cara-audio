"""
Long-form TTS Jobs API.

Handles batch/background TTS generation for long texts.
Uses Redis for job queue management.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from cara.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Models
# =============================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCreateRequest(BaseModel):
    """Request to create a long-form TTS job"""
    text: str = Field(..., description="Text to convert to speech", min_length=1)
    voice: str = Field(default="default", description="Voice ID")
    language: str = Field(default="it", description="Language code")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    format: str = Field(default="wav", pattern="^(wav|mp3)$")
    
    model_config = {"json_schema_extra": {"example": {
        "text": "Long text to be converted to speech...",
        "voice": "default",
        "language": "it",
        "speed": 1.0,
        "format": "wav"
    }}}


class JobResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: JobStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime
    text_length: int
    estimated_duration: float | None = None
    error: str | None = None
    audio_url: str | None = None


# =============================================================================
# In-Memory Job Store (Replace with Redis in production)
# =============================================================================

# Simple in-memory store for demonstration
# In production, use Redis or a database
_jobs: dict[str, dict] = {}


async def process_job(job_id: str, request: JobCreateRequest):
    """Background task to process a TTS job."""
    from cara.models import model_manager
    from cara.engines.longform import LongFormTTSEngine
    
    settings = get_settings()
    
    try:
        _jobs[job_id]["status"] = JobStatus.PROCESSING
        _jobs[job_id]["updated_at"] = datetime.utcnow()
        
        if not model_manager.is_loaded:
            await model_manager.load_all()
        
        engine = LongFormTTSEngine(model_manager.tts, settings)
        
        # Process text in chunks
        async for progress in engine.generate(
            text=request.text,
            job_id=job_id,
            voice=request.voice,
            language=request.language,
            speed=request.speed,
        ):
            _jobs[job_id]["progress"] = progress
            _jobs[job_id]["updated_at"] = datetime.utcnow()
        
        _jobs[job_id]["status"] = JobStatus.COMPLETED
        _jobs[job_id]["progress"] = 1.0
        _jobs[job_id]["audio_url"] = f"/tts/jobs/{job_id}/audio"
        _jobs[job_id]["updated_at"] = datetime.utcnow()
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        _jobs[job_id]["status"] = JobStatus.FAILED
        _jobs[job_id]["error"] = str(e)
        _jobs[job_id]["updated_at"] = datetime.utcnow()


# =============================================================================
# Endpoints
# =============================================================================

@router.post("", response_model=JobResponse)
async def create_job(request: JobCreateRequest, background_tasks: BackgroundTasks):
    """
    Create a new long-form TTS job.
    
    The job will be processed in the background.
    Poll the job status endpoint to check progress.
    """
    job_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    # Estimate duration (rough: ~150 words per minute)
    word_count = len(request.text.split())
    estimated_duration = (word_count / 150) * 60  # seconds
    
    _jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "progress": 0.0,
        "created_at": now,
        "updated_at": now,
        "text_length": len(request.text),
        "estimated_duration": estimated_duration,
        "error": None,
        "audio_url": None,
        "request": request.model_dump(),
    }
    
    # Start background processing
    background_tasks.add_task(process_job, job_id, request)
    
    return JobResponse(**_jobs[job_id])


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get the status of a TTS job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(**_jobs[job_id])


@router.get("/{job_id}/audio")
async def get_job_audio(job_id: str):
    """
    Download the generated audio for a completed job.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _jobs[job_id]
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Status: {job['status']}"
        )
    
    # TODO: Return actual audio file from storage
    # For now, return a placeholder
    raise HTTPException(
        status_code=501, 
        detail="Audio storage not yet implemented. Use streaming TTS for now."
    )


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending or processing job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _jobs[job_id]
    
    if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job['status']}"
        )
    
    _jobs[job_id]["status"] = JobStatus.CANCELLED
    _jobs[job_id]["updated_at"] = datetime.utcnow()
    
    return {"success": True, "job_id": job_id, "status": JobStatus.CANCELLED}


@router.get("")
async def list_jobs(
    status: Annotated[JobStatus | None, Query(description="Filter by status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """List all jobs, optionally filtered by status."""
    jobs = list(_jobs.values())
    
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    
    # Sort by created_at descending
    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    
    return {
        "jobs": [JobResponse(**j) for j in jobs[:limit]],
        "total": len(jobs),
    }

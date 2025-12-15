"""
Authentication middleware.

Simple API key authentication for protecting endpoints.
"""

from fastapi import Request, HTTPException
from fastapi.security import APIKeyHeader

from cara.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(request: Request) -> None:
    """
    Verify API key if configured.
    
    If no API key is set in settings, authentication is bypassed.
    """
    settings = get_settings()
    
    # Skip auth if no API key configured
    if not settings.server.api_key:
        return
    
    # Get API key from header
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key != settings.server.api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

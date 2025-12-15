# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Create virtual environment
RUN python3.11 -m venv /home/app/venv
ENV PATH="/home/app/venv/bin:$PATH"

# Install Python dependencies
COPY --chown=app:app pyproject.toml .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

# Copy application code
COPY --chown=app:app src/ ./src/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"

# Run the application
CMD ["uvicorn", "cara_audio.main:app", "--host", "0.0.0.0", "--port", "8000"]

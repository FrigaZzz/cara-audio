# CARA Audio Backend

GPU-accelerated TTS + STT service for CARA - the voice that takes care.

## Features

- 🎙️ **Text-to-Speech (TTS)**: Real-time streaming and long-form generation using XTTS-v2
- 🎧 **Speech-to-Text (STT)**: Fast transcription using Whisper (via faster-whisper)
- ⚡ **GPU Accelerated**: Optimized for NVIDIA GPUs with CUDA
- 🔧 **Easy Configuration**: Swap models and tune parameters via environment variables
- 🔄 **Hot Reload**: Change models at runtime without restart

## Quick Start

### Voice Cloning Setup

Place your reference audio files (e.g., `my_voice.wav`) in the `assets/voices/` directory.
These files are ignored by git to avoid uploading large binary files.
A default voice is provided at `assets/voices/default_speaker.wav`.

### 1. Install dependencies

```bash
# Using uv (recommended)
cd cara_audio_backend
uv sync

# Or using pip
pip install -e .
```

### 2. Configure environment

```bash
cp .env.template .env
# Edit .env with your settings
```

### 3. Run the server

```bash
# Development
conda activate cara-audio
uvicorn --app-dir src cara.main:app --reload

# Production
uv run uvicorn src.cara.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### TTS (Text-to-Speech)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tts/generate` | POST | Generate audio from text |
| `/tts/stream` | WebSocket | Real-time streaming TTS |
| `/tts/voices` | GET | List available voices |
| `/tts/reload` | POST | Hot-reload TTS model |

### STT (Speech-to-Text)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stt/transcribe` | POST | Transcribe audio file |
| `/stt/languages` | GET | List supported languages |
| `/stt/reload` | POST | Hot-reload STT model |

### Jobs (Long-form TTS)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tts/jobs` | POST | Create long-form TTS job |
| `/tts/jobs` | GET | List all jobs |
| `/tts/jobs/{id}` | GET | Get job status |
| `/tts/jobs/{id}/audio` | GET | Download completed audio |
| `/tts/jobs/{id}` | DELETE | Cancel job |

### Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Full health status |
| `/health/ready` | GET | Kubernetes readiness probe |
| `/health/live` | GET | Kubernetes liveness probe |
| `/metrics` | GET | Prometheus metrics |

## Configuration

All settings can be configured via environment variables:

### TTS Settings
- `TTS_MODEL_NAME`: Coqui TTS model name (default: `tts_models/multilingual/multi-dataset/xtts_v2`)
- `TTS_DEVICE`: Device for inference (`cuda` or `cpu`)
- `TTS_COMPUTE_TYPE`: Precision (`float16`, `float32`, `int8`)

### STT Settings
- `STT_MODEL_SIZE`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`)
- `STT_DEVICE`: Device for inference
- `STT_COMPUTE_TYPE`: Precision

### Audio Settings
- `AUDIO_SAMPLE_RATE`: Output sample rate (default: 24000)
- `STREAMING_CHUNK_SIZE`: Bytes per streaming chunk (default: 4096)

See `.env.template` for all available options.

## WebSocket Streaming Example

```javascript
const ws = new WebSocket('ws://localhost:8000/tts/stream');

ws.onopen = () => {
  ws.send(JSON.stringify({
    text: "Ciao Nonna, come stai oggi?",
    language: "it",
    speed: 1.0
  }));
};

ws.onmessage = (event) => {
  if (event.data instanceof Blob) {
    // Audio chunk - play it
    playAudioChunk(event.data);
  } else {
    const msg = JSON.parse(event.data);
    if (msg.done) {
      console.log(`Complete! Duration: ${msg.duration}s`);
    }
  }
};
```

## Development

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=cara_audio

# Format code
uv run ruff format .

# Lint
uv run ruff check .
```

## Docker

```bash
# Build
docker build -t cara-audio .

# Run with GPU
docker run --gpus all -p 8000:8000 cara-audio
```

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA 12+ (for GPU acceleration)
- ~8GB VRAM for XTTS-v2 + Whisper large-v3

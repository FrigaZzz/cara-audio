"""
Audio utilities for format conversion and processing.
"""

import io
import struct
from typing import Any

import numpy as np


def numpy_to_pcm16(audio_data: Any, normalize: bool = True) -> bytes:
    """
    Convert numpy audio array to PCM16 bytes.
    
    Args:
        audio_data: Numpy array of audio samples (float or int)
        normalize: Whether to normalize audio levels
        
    Returns:
        bytes: PCM16 audio data
    """
    # Ensure numpy array
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data)
    
    # Convert to float if needed
    if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
        audio_data = audio_data.astype(np.float32)
    
    # Normalize to [-1, 1] if needed
    if normalize:
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95  # Leave some headroom
    
    # Clip to valid range
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    # Convert to 16-bit integers
    pcm16 = (audio_data * 32767).astype(np.int16)
    
    return pcm16.tobytes()


def audio_to_wav_bytes(
    audio_data: Any,
    sample_rate: int = 24000,
    channels: int = 1,
) -> bytes:
    """
    Convert audio data to WAV format bytes.
    
    Args:
        audio_data: Numpy array of audio samples
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        
    Returns:
        bytes: Complete WAV file data
    """
    # Convert to PCM16
    pcm_data = numpy_to_pcm16(audio_data)
    
    # Build WAV file
    buffer = io.BytesIO()
    
    # Write WAV header
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    
    # RIFF header
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 36 + len(pcm_data)))  # File size - 8
    buffer.write(b'WAVE')
    
    # fmt chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # Chunk size
    buffer.write(struct.pack('<H', 1))   # Audio format (PCM)
    buffer.write(struct.pack('<H', channels))
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', byte_rate))
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', bits_per_sample))
    
    # data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<I', len(pcm_data)))
    buffer.write(pcm_data)
    
    return buffer.getvalue()


def resample_audio(
    audio_data: np.ndarray,
    original_rate: int,
    target_rate: int,
) -> np.ndarray:
    """
    Resample audio to a different sample rate.
    
    Args:
        audio_data: Numpy array of audio samples
        original_rate: Original sample rate
        target_rate: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if original_rate == target_rate:
        return audio_data
    
    # Use scipy for resampling
    from scipy import signal
    
    # Calculate resampling ratio
    ratio = target_rate / original_rate
    new_length = int(len(audio_data) * ratio)
    
    resampled = signal.resample(audio_data, new_length)
    return resampled.astype(audio_data.dtype)


def normalize_audio(audio_data: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to a target dB level.
    
    Args:
        audio_data: Numpy array of audio samples
        target_db: Target dB level (default -3 dB)
        
    Returns:
        Normalized audio array
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio_data ** 2))
    
    if rms == 0:
        return audio_data
    
    # Calculate target RMS
    target_rms = 10 ** (target_db / 20)
    
    # Scale audio
    return audio_data * (target_rms / rms)

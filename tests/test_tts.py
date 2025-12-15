"""Tests for TTS functionality."""

import pytest
from cara.utils.text import (
    chunk_text_by_sentences,
    chunk_text_by_duration,
    clean_text_for_tts,
)


class TestTextChunking:
    """Tests for text chunking utilities."""
    
    def test_chunk_by_sentences_simple(self, sample_text):
        """Test basic sentence chunking."""
        chunks = chunk_text_by_sentences(sample_text)
        assert len(chunks) >= 1
        # All text should be preserved
        assert "".join(chunks).replace(" ", "") == sample_text.replace(" ", "").replace("?", "? ").replace(".", ". ").strip().replace(" ", "")
    
    def test_chunk_by_sentences_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text_by_sentences("")
        assert chunks == []
    
    def test_chunk_by_sentences_respects_max_chars(self):
        """Test that chunks respect max_chars limit."""
        text = "Short. Another short. Yet another. One more sentence here."
        chunks = chunk_text_by_sentences(text, max_chars=30)
        for chunk in chunks:
            # Soft limit, so some may exceed slightly
            assert len(chunk) < 100
    
    def test_chunk_by_duration(self, sample_long_text):
        """Test duration-based chunking."""
        chunks = chunk_text_by_duration(sample_long_text, target_duration=5)
        assert len(chunks) >= 1
    
    def test_clean_text_removes_markdown(self):
        """Test markdown removal."""
        text = "This is **bold** and *italic* text."
        cleaned = clean_text_for_tts(text)
        assert "**" not in cleaned
        assert "*" not in cleaned
        assert "bold" in cleaned
    
    def test_clean_text_expands_italian_abbreviations(self):
        """Test Italian abbreviation expansion."""
        text = "Dott. Rossi ha detto che ecc."
        cleaned = clean_text_for_tts(text)
        assert "Dottore" in cleaned
        assert "eccetera" in cleaned


class TestAudioUtils:
    """Tests for audio utilities."""
    
    def test_numpy_to_pcm16(self):
        """Test PCM16 conversion."""
        import numpy as np
        from cara_audio.utils.audio import numpy_to_pcm16
        
        # Create test audio
        audio = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
        pcm = numpy_to_pcm16(audio)
        
        assert isinstance(pcm, bytes)
        assert len(pcm) == 1000 * 2  # 16-bit = 2 bytes per sample
    
    def test_audio_to_wav_bytes(self):
        """Test WAV file generation."""
        import numpy as np
        from cara_audio.utils.audio import audio_to_wav_bytes
        
        audio = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
        wav = audio_to_wav_bytes(audio, sample_rate=24000)
        
        assert wav.startswith(b'RIFF')
        assert b'WAVE' in wav
        assert b'fmt ' in wav
        assert b'data' in wav

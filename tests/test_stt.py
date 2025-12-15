"""Tests for STT functionality."""

import pytest


class TestTranscriptionEngine:
    """Tests for transcription engine."""
    
    @pytest.mark.integration
    async def test_transcribe_audio(self):
        """Integration test for audio transcription."""
        # This test requires models to be loaded
        # Skip if models not available
        pytest.skip("Integration test - requires GPU and models")

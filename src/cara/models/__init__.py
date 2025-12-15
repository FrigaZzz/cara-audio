"""Models package - TTS and STT model management"""

from cara.models.manager import ModelManager

# Global model manager instance
model_manager = ModelManager()

__all__ = ["model_manager"]

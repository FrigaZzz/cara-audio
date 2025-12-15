"""Test configuration and fixtures."""

import pytest


@pytest.fixture
def sample_text():
    """Sample Italian text for testing."""
    return "Ciao Nonna, come stai oggi? È ora della tua pillola per il cuore."


@pytest.fixture
def sample_long_text():
    """Longer text for long-form TTS testing."""
    return """
    Buongiorno cara Nonna! Oggi è una bella giornata di sole.
    Ricordati di prendere le tue medicine questa mattina.
    La pillola blu è per il cuore, quella bianca per la pressione.
    Dopo pranzo potresti fare una bella passeggiata in giardino.
    Ti voglio tanto bene!
    """

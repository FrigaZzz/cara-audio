"""
Text processing utilities for TTS.
"""

import re


def chunk_text_by_sentences(text: str, max_chars: int = 500) -> list[str]:
    """
    Split text into sentence-based chunks.
    
    This provides natural breaks for TTS generation,
    maintaining prosody across sentence boundaries.
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk (soft limit)
        
    Returns:
        List of text chunks
    """
    # Clean text
    text = text.strip()
    if not text:
        return []
    
    # Split by sentence-ending punctuation
    # Handle multiple languages (. ! ? and Italian-specific)
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    # Group sentences into chunks
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_length = len(sentence)
        
        # If adding this sentence exceeds limit, start new chunk
        if current_length + sentence_length > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length + 1  # +1 for space
    
    # Add remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def chunk_text_by_duration(
    text: str,
    target_duration: int = 20,
    words_per_second: float = 2.5,
) -> list[str]:
    """
    Split text into duration-based chunks for long-form TTS.
    
    Args:
        text: Input text to chunk
        target_duration: Target duration per chunk in seconds
        words_per_second: Estimated speaking rate
        
    Returns:
        List of text chunks
    """
    # Calculate target words per chunk
    target_words = int(target_duration * words_per_second)
    
    # First split by sentences
    sentences = chunk_text_by_sentences(text, max_chars=10000)  # Large limit
    
    chunks = []
    current_chunk = []
    current_words = 0
    
    for sentence in sentences:
        word_count = len(sentence.split())
        
        # If sentence alone exceeds target, add it as its own chunk
        if word_count >= target_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_words = 0
            chunks.append(sentence)
            continue
        
        # If adding this sentence exceeds target, start new chunk
        if current_words + word_count > target_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_words = 0
        
        current_chunk.append(sentence)
        current_words += word_count
    
    # Add remaining
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for TTS processing.
    
    - Normalize whitespace
    - Remove unspeakable characters
    - Expand common abbreviations
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove markdown/formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)       # Italic
    text = re.sub(r'`([^`]+)`', r'\1', text)         # Code
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
    
    # Remove special characters that shouldn't be spoken
    text = re.sub(r'[#@\[\]{}|\\]', '', text)
    
    # Italian abbreviations
    abbreviations = {
        r'\bDott\.\s': 'Dottore ',
        r'\bDott\.ssa\s': 'Dottoressa ',
        r'\bSig\.\s': 'Signor ',
        r'\bSig\.ra\s': 'Signora ',
        r'\bIng\.\s': 'Ingegner ',
        r'\bProf\.\s': 'Professor ',
        r'\becc\.': 'eccetera',
        r'\becc$': 'eccetera',
    }
    
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text.strip()

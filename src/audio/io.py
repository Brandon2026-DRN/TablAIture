"""
Audio I/O utilities for loading and saving audio files.
"""

import numpy as np
import librosa
from typing import Tuple, Optional


def load_audio(
    filepath: str,
    sr: Optional[int] = None,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file using librosa.
    
    Args:
        filepath: Path to audio file
        sr: Target sample rate (None = use original)
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sample_rate = librosa.load(
        filepath,
        sr=sr,
        mono=mono
    )
    return audio, sample_rate


def save_audio(
    audio: np.ndarray,
    filepath: str,
    sr: int
) -> None:
    """
    Save audio array to file.
    
    Args:
        audio: Audio signal array
        filepath: Output file path
        sr: Sample rate
    """
    import soundfile as sf
    sf.write(filepath, audio, sr)


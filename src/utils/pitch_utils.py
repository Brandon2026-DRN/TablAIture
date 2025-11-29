"""
Pitch and frequency utility functions.
"""

import numpy as np
from typing import Union


def hz_to_midi(freq: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert frequency in Hz to MIDI note number.
    
    Args:
        freq: Frequency in Hz
        
    Returns:
        MIDI note number (float)
    """
    return 69 + 12 * np.log2(freq / 440.0)


def midi_to_hz(midi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert MIDI note number to frequency in Hz.
    
    Args:
        midi: MIDI note number
        
    Returns:
        Frequency in Hz
    """
    return 440.0 * (2 ** ((midi - 69) / 12.0))


def pitch_class_to_name(pitch_class: int) -> str:
    """
    Convert pitch class (0-11) to note name.
    
    Args:
        pitch_class: Pitch class (0=C, 1=C#, ..., 11=B)
        
    Returns:
        Note name (e.g., "C", "C#", "D")
    """
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return names[pitch_class % 12]


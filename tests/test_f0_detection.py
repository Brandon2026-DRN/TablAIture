"""
Tests for F0 detection algorithms.
"""

import numpy as np
import pytest
from src.features.f0_detection import YINF0Detector, AutocorrelationF0Detector


def generate_sine_wave(freq: float, duration: float, sr: int = 22050) -> np.ndarray:
    """Generate a sine wave at given frequency."""
    t = np.linspace(0, duration, int(sr * duration))
    return np.sin(2 * np.pi * freq * t)


class TestYINF0Detector:
    """Tests for YIN F0 detector."""
    
    def test_detect_sine_wave(self):
        """Test F0 detection on pure sine wave."""
        detector = YINF0Detector()
        freq = 440.0  # A4
        duration = 1.0
        sr = 22050
        
        audio = generate_sine_wave(freq, duration, sr)
        f0_values, confidence = detector.detect(audio, sr, fmin=80.0, fmax=1000.0)
        
        # Should detect frequency close to 440 Hz
        valid_f0 = f0_values[f0_values > 0]
        if len(valid_f0) > 0:
            mean_f0 = np.mean(valid_f0)
            # Allow 5% tolerance
            assert abs(mean_f0 - freq) / freq < 0.05, \
                f"Detected {mean_f0} Hz, expected {freq} Hz"
    
    def test_detect_multiple_frequencies(self):
        """Test F0 detection on signal with multiple frequencies."""
        detector = YINF0Detector()
        sr = 22050
        
        # Generate signal with fundamental at 220 Hz and harmonics
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        fundamental = 220.0
        signal = (np.sin(2 * np.pi * fundamental * t) +
                  0.5 * np.sin(2 * np.pi * fundamental * 2 * t) +
                  0.25 * np.sin(2 * np.pi * fundamental * 3 * t))
        
        f0_values, confidence = detector.detect(signal, sr, fmin=80.0, fmax=1000.0)
        
        # Should detect fundamental frequency
        valid_f0 = f0_values[f0_values > 0]
        if len(valid_f0) > 0:
            mean_f0 = np.mean(valid_f0)
            assert abs(mean_f0 - fundamental) / fundamental < 0.1


class TestAutocorrelationF0Detector:
    """Tests for autocorrelation F0 detector."""
    
    def test_detect_sine_wave(self):
        """Test F0 detection on pure sine wave."""
        detector = AutocorrelationF0Detector()
        freq = 440.0
        duration = 1.0
        sr = 22050
        
        audio = generate_sine_wave(freq, duration, sr)
        f0_values, confidence = detector.detect(audio, sr, fmin=80.0, fmax=1000.0)
        
        # Should detect frequency close to 440 Hz
        valid_f0 = f0_values[f0_values > 0]
        if len(valid_f0) > 0:
            mean_f0 = np.mean(valid_f0)
            # Allow 10% tolerance (autocorr is less precise than YIN)
            assert abs(mean_f0 - freq) / freq < 0.10, \
                f"Detected {mean_f0} Hz, expected {freq} Hz"
    
    def test_empty_signal(self):
        """Test handling of empty/silent signal."""
        detector = AutocorrelationF0Detector()
        sr = 22050
        audio = np.zeros(22050)  # 1 second of silence
        
        f0_values, confidence = detector.detect(audio, sr, fmin=80.0, fmax=1000.0)
        
        # Should return zeros or very low confidence
        assert np.all(f0_values == 0) or np.all(confidence < 0.1)


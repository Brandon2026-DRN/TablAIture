"""
Tests for audio preprocessing.
"""

import numpy as np
import pytest
from src.audio.preprocessing import AudioPreprocessor


class TestAudioPreprocessor:
    """Tests for audio preprocessing."""
    
    def test_resample(self):
        """Test audio resampling."""
        preprocessor = AudioPreprocessor(target_sr=22050)
        
        # Generate test signal at 44100 Hz
        duration = 1.0
        sr_orig = 44100
        t = np.linspace(0, duration, int(sr_orig * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        resampled = preprocessor.resample(audio, sr_orig)
        
        # Check sample rate conversion
        assert len(resampled) == pytest.approx(22050, rel=0.01)
    
    def test_normalize_amplitude(self):
        """Test amplitude normalization."""
        preprocessor = AudioPreprocessor(normalize=True)
        
        # Generate signal with varying amplitude
        audio = np.random.randn(1000) * 0.5
        
        processed, _ = preprocessor.process(audio, 22050)
        
        # Check normalization
        assert np.max(np.abs(processed)) <= 1.0
        if np.max(np.abs(audio)) > 0:
            assert np.max(np.abs(processed)) > 0.9  # Should be close to 1.0
    
    def test_full_pipeline(self):
        """Test full preprocessing pipeline."""
        preprocessor = AudioPreprocessor(
            target_sr=22050,
            normalize=True,
            apply_denoising=True
        )
        
        # Generate test signal
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        
        processed, sr_out = preprocessor.process(audio, sr)
        
        # Check output
        assert sr_out == 22050
        assert len(processed) > 0
        assert np.max(np.abs(processed)) <= 1.0


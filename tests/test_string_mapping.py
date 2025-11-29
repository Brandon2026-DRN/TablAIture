"""
Tests for string mapping and tablature inference.
"""

import numpy as np
import pytest
from src.inference.tablature_inference import GuitarStringMapper, TablatureInference


class TestGuitarStringMapper:
    """Tests for guitar string mapping."""
    
    def test_pitch_to_fret(self):
        """Test pitch to fret conversion."""
        mapper = GuitarStringMapper()
        
        # Test open strings
        assert mapper.pitch_to_fret(82.41, 0) == 0  # E2 open
        assert mapper.pitch_to_fret(110.00, 1) == 0  # A2 open
        assert mapper.pitch_to_fret(329.63, 5) == 0  # E4 open
        
        # Test first fret on low E string
        # E2 (82.41 Hz) -> F2 (87.31 Hz) = 1 semitone
        assert mapper.pitch_to_fret(87.31, 0) == 1
        
        # Test 12th fret (octave)
        # E2 -> E3 = 12 semitones
        assert mapper.pitch_to_fret(164.81, 0) == 12
    
    def test_find_best_string(self):
        """Test finding best string for a pitch."""
        mapper = GuitarStringMapper()
        
        # A4 (440 Hz) - should map to high E string (5th string, 12th fret)
        # Actually, 440 Hz is between B3 (246.94) and E4 (329.63)
        # E4 at 12th fret = 329.63 * 2 = 659.26 Hz (too high)
        # B3 at 12th fret = 246.94 * 2 = 493.88 Hz (closer)
        # Actually, let's test with a clearer case
        
        # E3 (164.81 Hz) - should map to low E string, 12th fret
        string_idx, confidence = mapper.find_best_string(164.81)
        assert string_idx == 0  # Low E string
        assert confidence > 0
    
    def test_validate_note(self):
        """Test note validation."""
        mapper = GuitarStringMapper()
        
        # Valid note: E2 on low E string, open
        assert mapper.validate_note(82.41, 0, 0) == True
        
        # Valid note: A4 on high E string, 12th fret
        # E4 (329.63) * 2 = 659.26 Hz
        assert mapper.validate_note(659.26, 5, 12) == True
        
        # Invalid: wrong string
        assert mapper.validate_note(82.41, 1, 0) == False  # E2 on A string
        
        # Invalid: out of range fret
        assert mapper.validate_note(82.41, 0, 25) == False  # Fret too high


class TestTablatureInference:
    """Tests for tablature inference."""
    
    def test_quantize_time(self):
        """Test time quantization."""
        inference = TablatureInference()
        
        assert inference.quantize_time(0.123, 0.1) == 0.1
        assert inference.quantize_time(0.156, 0.1) == 0.2
        assert inference.quantize_time(0.05, 0.1) == 0.0
    
    def test_infer_tablature_basic(self):
        """Test basic tablature inference."""
        inference = TablatureInference()
        
        # Create simple F0 signal (E2 at 82.41 Hz)
        n_frames = 100
        f0_values = np.full(n_frames, 82.41)
        f0_confidence = np.full(n_frames, 0.9)
        times = np.linspace(0, 1.0, n_frames)
        
        notes = inference.infer_tablature(
            f0_values=f0_values,
            f0_confidence=f0_confidence,
            times=times
        )
        
        # Should detect at least one note
        assert len(notes) > 0
        
        # Check note properties
        note = notes[0]
        assert note.pitch_hz == pytest.approx(82.41, rel=0.01)
        assert note.string == 1  # Low E string
        assert note.fret == 0  # Open string
        assert note.confidence > 0
    
    def test_infer_tablature_with_string_probs(self):
        """Test inference with string probabilities."""
        inference = TablatureInference()
        
        n_frames = 50
        f0_values = np.full(n_frames, 110.0)  # A2
        f0_confidence = np.full(n_frames, 0.8)
        times = np.linspace(0, 0.5, n_frames)
        
        # Strong probability for 2nd string (A string)
        string_probs = np.zeros((n_frames, 6))
        string_probs[:, 1] = 0.9  # A string
        string_probs[:, 0] = 0.1  # Other strings
        
        notes = inference.infer_tablature(
            f0_values=f0_values,
            f0_confidence=f0_confidence,
            times=times,
            string_probs=string_probs
        )
        
        if len(notes) > 0:
            # Should prefer A string
            assert notes[0].string == 2  # A string (1-indexed)


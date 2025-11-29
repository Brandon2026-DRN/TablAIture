"""
Tablature inference: converts detected notes to guitar string and fret positions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NoteEvent:
    """Represents a single note event in tablature."""
    time: float
    string: int
    fret: int
    pitch_hz: float
    confidence: float


class GuitarStringMapper:
    """
    Maps pitch frequencies to guitar string and fret positions.
    
    Standard tuning: E2 (82.41 Hz), A2 (110.00 Hz), D3 (146.83 Hz),
                     G3 (196.00 Hz), B3 (246.94 Hz), E4 (329.63 Hz)
    """
    
    # Standard guitar tuning (low to high)
    OPEN_STRINGS = [
        82.41,   # E2 (6th string)
        110.00,  # A2 (5th string)
        146.83,  # D3 (4th string)
        196.00,  # G3 (3rd string)
        246.94,  # B3 (2nd string)
        329.63   # E4 (1st string)
    ]
    
    # Maximum fret (typically 24)
    MAX_FRET = 24
    
    def __init__(self, tuning: Optional[List[float]] = None):
        """
        Initialize string mapper.
        
        Args:
            tuning: Custom tuning frequencies (default: standard EADGBE)
        """
        self.tuning = tuning if tuning is not None else self.OPEN_STRINGS.copy()
        self.n_strings = len(self.tuning)
    
    def pitch_to_fret(self, pitch_hz: float, string_idx: int) -> Optional[int]:
        """
        Convert pitch frequency to fret number on a given string.
        
        Args:
            pitch_hz: Pitch frequency in Hz
            string_idx: String index (0-5, 0 = lowest string)
            
        Returns:
            Fret number (0 = open, None if out of range)
        """
        if string_idx < 0 or string_idx >= self.n_strings:
            return None
        
        open_freq = self.tuning[string_idx]
        
        # Calculate semitone difference
        semitones = 12 * np.log2(pitch_hz / open_freq)
        fret = int(round(semitones))
        
        # Validate fret range
        if fret < 0 or fret > self.MAX_FRET:
            return None
        
        return fret
    
    def find_best_string(
        self,
        pitch_hz: float,
        string_probs: Optional[np.ndarray] = None
    ) -> Tuple[int, float]:
        """
        Find the best string assignment for a given pitch.
        
        Args:
            pitch_hz: Pitch frequency in Hz
            string_probs: Optional string probability distribution [6]
            
        Returns:
            Tuple of (string_idx, confidence)
        """
        best_string = 0
        best_score = -np.inf
        
        for string_idx in range(self.n_strings):
            fret = self.pitch_to_fret(pitch_hz, string_idx)
            
            if fret is None:
                continue
            
            # Score based on fret position (prefer lower frets) and string prob
            fret_score = 1.0 / (fret + 1)  # Prefer lower frets
            
            if string_probs is not None:
                string_score = string_probs[string_idx]
            else:
                string_score = 1.0
            
            total_score = fret_score * string_score
            
            if total_score > best_score:
                best_score = total_score
                best_string = string_idx
        
        confidence = min(best_score, 1.0)
        return best_string, confidence
    
    def validate_note(
        self,
        pitch_hz: float,
        string_idx: int,
        fret: int
    ) -> bool:
        """
        Validate that a note assignment is physically possible.
        
        Args:
            pitch_hz: Pitch frequency in Hz
            string_idx: String index
            fret: Fret number
            
        Returns:
            True if assignment is valid
        """
        if string_idx < 0 or string_idx >= self.n_strings:
            return False
        
        if fret < 0 or fret > self.MAX_FRET:
            return False
        
        # Calculate expected frequency
        open_freq = self.tuning[string_idx]
        expected_freq = open_freq * (2 ** (fret / 12.0))
        
        # Allow small tolerance (within 50 cents)
        freq_ratio = pitch_hz / expected_freq
        semitone_diff = 12 * np.log2(freq_ratio)
        
        return abs(semitone_diff) < 0.5


class TablatureInference:
    """
    Main inference engine that converts detected notes to tablature.
    
    Combines F0 detection, neural classification, and string mapping.
    """
    
    def __init__(
        self,
        string_mapper: Optional[GuitarStringMapper] = None,
        min_confidence: float = 0.3,
        min_note_duration: float = 0.05
    ):
        """
        Initialize tablature inference engine.
        
        Args:
            string_mapper: String mapper instance (None = create default)
            min_confidence: Minimum confidence threshold for notes
            min_note_duration: Minimum note duration in seconds
        """
        self.string_mapper = string_mapper or GuitarStringMapper()
        self.min_confidence = min_confidence
        self.min_note_duration = min_note_duration
    
    def quantize_time(self, time: float, resolution: float = 0.1) -> float:
        """
        Quantize time to nearest resolution.
        
        Args:
            time: Time in seconds
            resolution: Time resolution in seconds (default: 0.1 = 100ms)
            
        Returns:
            Quantized time
        """
        return round(time / resolution) * resolution
    
    def detect_onsets(
        self,
        onset_probs: np.ndarray,
        times: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Detect note onsets from onset probabilities.
        
        Args:
            onset_probs: Onset probabilities [n_frames, 2]
            times: Time array for each frame
            threshold: Onset detection threshold
            
        Returns:
            Array of onset times
        """
        onset_scores = onset_probs[:, 1]  # Probability of onset
        onset_mask = onset_scores > threshold
        
        # Find onset indices
        onset_indices = np.where(onset_mask)[0]
        
        # Filter consecutive onsets (keep only first in cluster)
        if len(onset_indices) > 0:
            filtered_indices = [onset_indices[0]]
            for i in range(1, len(onset_indices)):
                if onset_indices[i] - filtered_indices[-1] > 2:  # At least 2 frames apart
                    filtered_indices.append(onset_indices[i])
            onset_indices = np.array(filtered_indices)
        
        return times[onset_indices] if len(onset_indices) > 0 else np.array([])
    
    def infer_tablature(
        self,
        f0_values: np.ndarray,
        f0_confidence: np.ndarray,
        times: np.ndarray,
        string_probs: Optional[np.ndarray] = None,
        onset_probs: Optional[np.ndarray] = None
    ) -> List[NoteEvent]:
        """
        Infer tablature from F0 detection and optional neural predictions.
        
        Args:
            f0_values: F0 values in Hz [n_frames]
            f0_confidence: F0 confidence scores [n_frames]
            times: Time array for each frame [n_frames]
            string_probs: Optional string probabilities [n_frames, 6]
            onset_probs: Optional onset probabilities [n_frames, 2]
            
        Returns:
            List of NoteEvent objects
        """
        notes = []
        
        # Detect onsets if available
        if onset_probs is not None:
            onset_times = self.detect_onsets(onset_probs, times)
        else:
            # Fallback: use F0 confidence peaks
            onset_mask = f0_confidence > self.min_confidence
            onset_indices = np.where(onset_mask)[0]
            if len(onset_indices) > 0:
                # Simple peak detection
                peaks = []
                for i in range(1, len(onset_indices) - 1):
                    if (f0_confidence[onset_indices[i]] > 
                        f0_confidence[onset_indices[i-1]] and
                        f0_confidence[onset_indices[i]] > 
                        f0_confidence[onset_indices[i+1]]):
                        peaks.append(onset_indices[i])
                onset_times = times[peaks] if peaks else times[onset_indices]
            else:
                onset_times = np.array([])
        
        # Process each frame with valid F0
        for i, (f0, conf, time) in enumerate(zip(f0_values, f0_confidence, times)):
            if f0 <= 0 or conf < self.min_confidence:
                continue
            
            # Get string probabilities for this frame
            frame_string_probs = None
            if string_probs is not None:
                frame_string_probs = string_probs[i]
            
            # Find best string assignment
            string_idx, string_conf = self.string_mapper.find_best_string(
                f0,
                frame_string_probs
            )
            
            # Calculate fret
            fret = self.string_mapper.pitch_to_fret(f0, string_idx)
            if fret is None:
                continue
            
            # Validate note
            if not self.string_mapper.validate_note(f0, string_idx, fret):
                continue
            
            # Combine confidences
            combined_confidence = conf * string_conf
            
            # Quantize time
            quantized_time = self.quantize_time(time)
            
            # Create note event
            note = NoteEvent(
                time=quantized_time,
                string=string_idx + 1,  # 1-indexed for output
                fret=fret,
                pitch_hz=f0,
                confidence=combined_confidence
            )
            
            notes.append(note)
        
        # Remove duplicate notes at same time
        notes = self._deduplicate_notes(notes)
        
        # Filter by minimum duration
        notes = self._filter_short_notes(notes)
        
        return notes
    
    def _deduplicate_notes(self, notes: List[NoteEvent]) -> List[NoteEvent]:
        """Remove duplicate notes at the same time, keeping highest confidence."""
        if not notes:
            return notes
        
        # Group by quantized time
        time_groups: Dict[float, List[NoteEvent]] = {}
        for note in notes:
            key = note.time
            if key not in time_groups:
                time_groups[key] = []
            time_groups[key].append(note)
        
        # Keep best note per time
        deduplicated = []
        for time_key, note_list in time_groups.items():
            # Sort by confidence and keep best
            note_list.sort(key=lambda n: n.confidence, reverse=True)
            deduplicated.append(note_list[0])
        
        # Sort by time
        deduplicated.sort(key=lambda n: n.time)
        return deduplicated
    
    def _filter_short_notes(
        self,
        notes: List[NoteEvent]
    ) -> List[NoteEvent]:
        """Filter out notes that are too short."""
        if len(notes) < 2:
            return notes
        
        filtered = []
        for i, note in enumerate(notes):
            # Check duration to next note
            if i < len(notes) - 1:
                duration = notes[i + 1].time - note.time
            else:
                duration = self.min_note_duration  # Last note, assume minimum
            
            if duration >= self.min_note_duration:
                filtered.append(note)
        
        return filtered


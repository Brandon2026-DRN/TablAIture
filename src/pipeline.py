"""
Main pipeline orchestrator for TablAIture system.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .audio.preprocessing import AudioPreprocessor
from .audio.io import load_audio
from .features.spectrogram import SpectrogramGenerator
from .features.f0_detection import F0Detector, YINF0Detector
from .models.note_classifier import NoteClassifier
from .inference.tablature_inference import TablatureInference, NoteEvent
from .utils.formatting import notes_to_json, format_plain_text_tab, format_musicxml


class TablAIturePipeline:
    """
    End-to-end pipeline for converting audio to tablature.
    
    Orchestrates all stages:
    1. Audio preprocessing
    2. Spectrogram generation
    3. F0 detection
    4. Neural note classification (optional)
    5. Tablature inference
    6. Formatting
    """
    
    def __init__(
        self,
        target_sr: int = 22050,
        use_neural_model: bool = False,
        f0_detector: Optional[F0Detector] = None,
        device: str = "cpu"
    ):
        """
        Initialize pipeline.
        
        Args:
            target_sr: Target sample rate for processing
            use_neural_model: Whether to use neural note classifier
            f0_detector: F0 detector instance (None = use YIN)
            device: Device for neural models ('cpu' or 'cuda')
        """
        # Initialize components
        self.preprocessor = AudioPreprocessor(target_sr=target_sr)
        self.spectrogram_gen = SpectrogramGenerator()
        self.f0_detector = f0_detector or YINF0Detector()
        self.use_neural_model = use_neural_model
        
        if use_neural_model:
            self.note_classifier = NoteClassifier(device=device)
        else:
            self.note_classifier = None
        
        self.tablature_inference = TablatureInference()
    
    def process_audio_file(
        self,
        audio_path: str,
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Process audio file and generate tablature.
        
        Args:
            audio_path: Path to input audio file
            output_format: Output format ('json', 'plain_text', 'musicxml', or 'all')
            
        Returns:
            Dictionary containing tablature in requested format(s)
        """
        # Load audio
        audio, sr = load_audio(audio_path)
        
        # Process through pipeline
        notes = self.process_audio(audio, sr)
        
        # Format output
        result = {}
        
        if output_format in ["json", "all"]:
            result["json"] = notes_to_json(notes)
        
        if output_format in ["plain_text", "all"]:
            result["plain_text"] = format_plain_text_tab(notes)
        
        if output_format in ["musicxml", "all"]:
            result["musicxml"] = format_musicxml(notes)
        
        if output_format == "json" and len(result) == 1:
            return result["json"]
        
        return result
    
    def process_audio(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[NoteEvent]:
        """
        Process audio array through full pipeline.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            List of NoteEvent objects
        """
        # Stage 1: Preprocessing
        audio_processed, sr_processed = self.preprocessor.process(audio, sr)
        
        # Stage 2: Spectrogram generation
        spectrogram, phase = self.spectrogram_gen.generate(audio_processed, sr_processed)
        times = self.spectrogram_gen.get_time_frames(
            len(audio_processed),
            sr_processed
        )
        
        # Stage 3: F0 detection
        f0_values, f0_confidence = self.f0_detector.detect(
            audio_processed,
            sr_processed,
            fmin=80.0,
            fmax=1000.0
        )
        
        # Ensure F0 arrays match time frames
        if len(f0_values) != len(times):
            # Interpolate or truncate to match
            min_len = min(len(f0_values), len(times))
            f0_values = f0_values[:min_len]
            f0_confidence = f0_confidence[:min_len]
            times = times[:min_len]
        
        # Stage 4: Neural note classification (optional)
        string_probs = None
        onset_probs = None
        
        if self.use_neural_model and self.note_classifier is not None:
            try:
                pitch_probs, string_probs, onset_probs = self.note_classifier.predict(
                    spectrogram
                )
            except Exception as e:
                print(f"Warning: Neural model prediction failed: {e}")
                print("Falling back to F0-only inference")
        
        # Stage 5: Tablature inference
        notes = self.tablature_inference.infer_tablature(
            f0_values=f0_values,
            f0_confidence=f0_confidence,
            times=times,
            string_probs=string_probs,
            onset_probs=onset_probs
        )
        
        return notes
    
    def process_batch(
        self,
        audio_paths: List[str],
        output_format: str = "json"
    ) -> List[Dict[str, Any]]:
        """
        Process multiple audio files in batch.
        
        Args:
            audio_paths: List of audio file paths
            output_format: Output format for each file
            
        Returns:
            List of results, one per input file
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.process_audio_file(audio_path, output_format)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({"error": str(e)})
        
        return results


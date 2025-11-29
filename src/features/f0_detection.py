"""
Fundamental frequency (F0) detection using YIN, autocorrelation, and peak picking.
"""

import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod
import librosa


class F0Detector(ABC):
    """Base class for F0 detection algorithms."""
    
    @abstractmethod
    def detect(
        self,
        audio: np.ndarray,
        sr: int,
        fmin: float = 80.0,
        fmax: float = 1000.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect fundamental frequencies.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
            
        Returns:
            Tuple of (f0_values, confidence_scores) where f0_values[i] is the
            F0 at frame i, and confidence_scores[i] is the confidence [0, 1]
        """
        pass


class YINF0Detector(F0Detector):
    """
    YIN algorithm for fundamental frequency detection.
    
    YIN is a time-domain autocorrelation-based method that is robust to
    harmonics and noise.
    """
    
    def __init__(self, frame_length: int = 2048, hop_length: int = 512):
        """
        Initialize YIN F0 detector.
        
        Args:
            frame_length: Analysis frame length in samples
            hop_length: Hop length between frames
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def detect(
        self,
        audio: np.ndarray,
        sr: int,
        fmin: float = 80.0,
        fmax: float = 1000.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect F0 using YIN algorithm.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
            
        Returns:
            Tuple of (f0_values, confidence_scores)
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=fmin,
            fmax=fmax,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        # Convert NaN to 0 for unvoiced frames
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # Use voiced_probs as confidence
        confidence = voiced_probs
        
        return f0, confidence


class AutocorrelationF0Detector(F0Detector):
    """
    Autocorrelation-based F0 detection.
    
    Uses autocorrelation to find periodic patterns in the signal.
    """
    
    def __init__(self, frame_length: int = 2048, hop_length: int = 512):
        """
        Initialize autocorrelation F0 detector.
        
        Args:
            frame_length: Analysis frame length in samples
            hop_length: Hop length between frames
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def _autocorr(self, frame: np.ndarray) -> np.ndarray:
        """Compute autocorrelation of a frame."""
        # Normalize frame
        frame = frame - np.mean(frame)
        frame = frame / (np.std(frame) + 1e-8)
        
        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        
        return autocorr
    
    def _find_peak(
        self,
        autocorr: np.ndarray,
        sr: int,
        fmin: float,
        fmax: float
    ) -> Tuple[float, float]:
        """
        Find peak in autocorrelation corresponding to F0.
        
        Returns:
            Tuple of (f0, confidence)
        """
        # Find valid lag range
        min_lag = int(sr / fmax)
        max_lag = int(sr / fmin)
        max_lag = min(max_lag, len(autocorr) - 1)
        
        if min_lag >= max_lag:
            return 0.0, 0.0
        
        # Search for peak in valid range
        search_range = autocorr[min_lag:max_lag + 1]
        if len(search_range) == 0:
            return 0.0, 0.0
        
        peak_idx = np.argmax(search_range) + min_lag
        peak_value = autocorr[peak_idx]
        
        # Convert lag to frequency
        if peak_idx > 0:
            f0 = sr / peak_idx
        else:
            return 0.0, 0.0
        
        # Confidence based on peak strength relative to autocorr[0]
        confidence = peak_value / (autocorr[0] + 1e-8)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Validate frequency range
        if f0 < fmin or f0 > fmax:
            return 0.0, 0.0
        
        return f0, confidence
    
    def detect(
        self,
        audio: np.ndarray,
        sr: int,
        fmin: float = 80.0,
        fmax: float = 1000.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect F0 using autocorrelation.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
            
        Returns:
            Tuple of (f0_values, confidence_scores)
        """
        n_frames = int(np.ceil((len(audio) - self.frame_length) / self.hop_length)) + 1
        f0_values = np.zeros(n_frames)
        confidence_scores = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            
            if end > len(audio):
                frame = np.pad(audio[start:], (0, end - len(audio)))
            else:
                frame = audio[start:end]
            
            autocorr = self._autocorr(frame)
            f0, confidence = self._find_peak(autocorr, sr, fmin, fmax)
            
            f0_values[i] = f0
            confidence_scores[i] = confidence
        
        return f0_values, confidence_scores


class PeakPickingF0Detector(F0Detector):
    """
    F0 detection using peak picking on spectrogram.
    
    Finds the strongest peak in each frequency bin and estimates F0.
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        threshold: float = 0.1
    ):
        """
        Initialize peak-picking F0 detector.
        
        Args:
            n_fft: FFT window size
            hop_length: Hop length between frames
            threshold: Minimum magnitude threshold for peak detection
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.threshold = threshold
    
    def detect(
        self,
        audio: np.ndarray,
        sr: int,
        fmin: float = 80.0,
        fmax: float = 1000.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect F0 using peak picking on spectrogram.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
            
        Returns:
            Tuple of (f0_values, confidence_scores)
        """
        # Compute spectrogram
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        # Find valid frequency range
        min_bin = np.argmax(freqs >= fmin)
        max_bin = np.argmax(freqs >= fmax)
        if max_bin == 0:
            max_bin = len(freqs)
        
        n_frames = magnitude.shape[1]
        f0_values = np.zeros(n_frames)
        confidence_scores = np.zeros(n_frames)
        
        # Normalize magnitude
        magnitude_norm = magnitude / (np.max(magnitude) + 1e-8)
        
        for frame_idx in range(n_frames):
            frame_magnitude = magnitude_norm[min_bin:max_bin, frame_idx]
            
            if np.max(frame_magnitude) < self.threshold:
                f0_values[frame_idx] = 0.0
                confidence_scores[frame_idx] = 0.0
                continue
            
            # Find peak
            peak_bin = np.argmax(frame_magnitude) + min_bin
            peak_freq = freqs[peak_bin]
            peak_magnitude = magnitude_norm[peak_bin, frame_idx]
            
            f0_values[frame_idx] = peak_freq
            confidence_scores[frame_idx] = peak_magnitude
        
        return f0_values, confidence_scores


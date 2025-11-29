"""
Audio preprocessing module for resampling, denoising, and normalization.
"""

import numpy as np
import librosa
from typing import Tuple, Optional
from scipy import signal


class AudioPreprocessor:
    """
    Preprocesses raw audio for tablature generation.
    
    Handles:
    - Resampling to consistent sample rate
    - Denoising/filtering
    - Amplitude normalization
    """
    
    def __init__(
        self,
        target_sr: int = 22050,
        normalize: bool = True,
        apply_denoising: bool = True,
        highpass_cutoff: float = 80.0,
        lowpass_cutoff: Optional[float] = None
    ):
        """
        Initialize audio preprocessor.
        
        Args:
            target_sr: Target sample rate in Hz (default: 22050)
            normalize: Whether to normalize amplitude to [-1, 1]
            apply_denoising: Whether to apply basic denoising filter
            highpass_cutoff: High-pass filter cutoff in Hz (removes low-frequency noise)
            lowpass_cutoff: Low-pass filter cutoff in Hz (None = no filtering)
        """
        self.target_sr = target_sr
        self.normalize = normalize
        self.apply_denoising = apply_denoising
        self.highpass_cutoff = highpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
    
    def resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Input audio signal
            sr: Original sample rate
            
        Returns:
            Resampled audio signal
        """
        if sr == self.target_sr:
            return audio
        
        return librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=self.target_sr,
            res_type="kaiser_best"
        )
    
    def apply_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply high-pass and optional low-pass filters.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Filtered audio signal
        """
        nyquist = self.target_sr / 2.0
        
        # High-pass filter to remove low-frequency noise
        if self.highpass_cutoff > 0 and self.highpass_cutoff < nyquist:
            sos = signal.butter(
                4,
                self.highpass_cutoff / nyquist,
                btype="high",
                output="sos"
            )
            audio = signal.sosfilt(sos, audio)
        
        # Low-pass filter to remove high-frequency noise
        if self.lowpass_cutoff is not None and self.lowpass_cutoff < nyquist:
            sos = signal.butter(
                4,
                self.lowpass_cutoff / nyquist,
                btype="low",
                output="sos"
            )
            audio = signal.sosfilt(sos, audio)
        
        return audio
    
    def denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply basic spectral subtraction denoising.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Denoised audio signal
        """
        # Simple spectral subtraction
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise floor from first 0.5 seconds
        noise_frames = int(0.5 * self.target_sr / 512)
        noise_floor = np.median(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Subtract noise floor with over-subtraction factor
        alpha = 2.0
        magnitude_denoised = magnitude - alpha * noise_floor
        magnitude_denoised = np.maximum(magnitude_denoised, 0.1 * magnitude)
        
        # Reconstruct signal
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(stft_denoised, hop_length=512)
        
        return audio_denoised
    
    def normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude to [-1, 1] range.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Normalized audio signal
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def process(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """
        Apply full preprocessing pipeline.
        
        Args:
            audio: Input audio signal
            sr: Original sample rate
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        # Resample
        audio = self.resample(audio, sr)
        
        # Apply filters
        audio = self.apply_filter(audio)
        
        # Denoise
        if self.apply_denoising:
            audio = self.denoise(audio)
        
        # Normalize
        if self.normalize:
            audio = self.normalize_amplitude(audio)
        
        return audio, self.target_sr


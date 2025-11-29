"""
Spectrogram generation module using STFT and Mel-spectrograms.
"""

import numpy as np
import librosa
from typing import Tuple, Optional


class SpectrogramGenerator:
    """
    Generates time-frequency representations for neural models.
    
    Supports:
    - Short-Time Fourier Transform (STFT)
    - Mel-spectrograms
    - Log-magnitude scaling
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 80.0,
        fmax: Optional[float] = None,
        use_mel: bool = True,
        log_scale: bool = True
    ):
        """
        Initialize spectrogram generator.
        
        Args:
            n_fft: FFT window size
            hop_length: Number of samples between frames
            n_mels: Number of mel filter banks (for mel-spectrogram)
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz (None = Nyquist)
            use_mel: Use mel-spectrogram instead of linear STFT
            log_scale: Apply log scaling to magnitude
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.use_mel = use_mel
        self.log_scale = log_scale
    
    def compute_stft(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute STFT spectrogram.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Tuple of (magnitude_spectrogram, phase_spectrogram)
        """
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window="hann"
        )
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        if self.log_scale:
            magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        return magnitude, phase
    
    def compute_mel_spectrogram(
        self,
        audio: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
        Compute mel-spectrogram.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Mel-spectrogram array
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            window="hann"
        )
        
        if self.log_scale:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec
    
    def generate(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate spectrogram representation.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            
        Returns:
            Tuple of (spectrogram, phase) where phase is None for mel-spectrograms
        """
        if self.use_mel:
            spec = self.compute_mel_spectrogram(audio, sr)
            return spec, None
        else:
            magnitude, phase = self.compute_stft(audio, sr)
            return magnitude, phase
    
    def get_time_frames(self, audio_length: int, sr: int) -> np.ndarray:
        """
        Get time axis for spectrogram frames.
        
        Args:
            audio_length: Length of audio in samples
            sr: Sample rate
            
        Returns:
            Array of time values in seconds for each frame
        """
        n_frames = int(np.ceil(audio_length / self.hop_length))
        frame_times = librosa.frames_to_time(
            np.arange(n_frames),
            sr=sr,
            hop_length=self.hop_length
        )
        return frame_times
    
    def get_frequency_bins(self, sr: int) -> np.ndarray:
        """
        Get frequency axis for spectrogram bins.
        
        Args:
            sr: Sample rate
            
        Returns:
            Array of frequency values in Hz for each bin
        """
        if self.use_mel:
            # Mel frequencies
            freqs = librosa.mel_frequencies(
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax or (sr / 2)
            )
        else:
            # Linear FFT frequencies
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        return freqs


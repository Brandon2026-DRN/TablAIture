# TablAIture

An end-to-end AI-powered system that converts raw guitar audio into accurate guitar tablature using a deep, multi-stage processing pipeline. Previously attained upwards of 5 Pull req & 15 stars, but privated due to Open-Sourced concerns.

## Features

- **Multi-Stage Pipeline**: Preprocessing to Spectrogram Generation to F0 Detection to Neural Classification to Tablature Inference to Formatting (structure)
- **Multiple F0 Detection Methods**: YIN algorithm, autocorrelation, and peak picking
- **Neural Note Classification**: CNN-based models for pitch, string, and onset detection
- **Flexible Output Formats**: JSON, plain-text tablature, and MusicXML
- **Modular Architecture**: Clean, well-documented codebase with type hints

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd TABLAITURE

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

```bash
# Basic usage
python -m src.main input_audio.wav -o output.json

# Specify output format
python -m src.main input_audio.wav -f plain_text -o output.txt

# Use autocorrelation F0 detection
python -m src.main input_audio.wav --f0-method autocorr -o output.json

# Use neural model (requires trained checkpoint)
python -m src.main input_audio.wav --use-neural -o output.json
```

### Python API

```python
from src.pipeline import TablAIturePipeline

# Initialize pipeline
pipeline = TablAIturePipeline()

# Process audio file
result = pipeline.process_audio_file("guitar_audio.wav", output_format="json")

# Result is a list of note events:
# [
#   {
#     "time": 0.0,
#     "string": 1,
#     "fret": 0,
#     "pitch_hz": 82.41,
#     "confidence": 0.95
#   },
#   ...
# ]
```

## Architecture

### Pipeline Stages

1. **Preprocessing**
   - Resampling to consistent sample rate (default: 22050 Hz)
   - High-pass filtering to remove low-frequency noise
   - Spectral subtraction denoising
   - Amplitude normalization

2. **Spectrogram Generation**
   - STFT or Mel-spectrogram computation
   - Configurable FFT parameters
   - Log-magnitude scaling

3. **Frequency-Domain Note Detection**
   - YIN algorithm for robust F0 estimation
   - Autocorrelation-based detection
   - Peak picking on spectrograms

4. **Neural Note Classification** (Optional)
   - CNN-based pitch class classification
   - String assignment probability estimation
   - Onset/offset detection

5. **Tablature Inference**
   - Pitch-to-string mapping using guitar tuning
   - Fret calculation from frequency
   - Temporal quantization and note deduplication

6. **Formatting**
   - JSON output (required format)
   - Plain-text tablature
   - MusicXML export

## Project Structure

```
TABLAITURE/
├── src/
│   ├── audio/           # Audio I/O and preprocessing
│   ├── features/        # Spectrogram and F0 detection
│   ├── models/          # Neural network models
│   ├── inference/       # Tablature inference engine
│   ├── utils/           # Utility functions
│   ├── pipeline.py      # Main pipeline orchestrator
│   └── main.py          # CLI interface
├── tests/               # Test suite
├── notebooks/           # Jupyter notebooks for experimentation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Output Format

The system outputs tablature in the following JSON format:

```json
[
  {
    "time": 0.0,
    "string": 1,
    "fret": 0,
    "pitch_hz": 82.41,
    "confidence": 0.95
  },
  {
    "time": 0.5,
    "string": 2,
    "fret": 3,
    "pitch_hz": 146.83,
    "confidence": 0.87
  }
]
```

Where:
- `time`: Time in seconds
- `string`: Guitar string (1-6, 1 = lowest E string)
- `fret`: Fret number (0 = open string)
- `pitch_hz`: Detected pitch frequency in Hz
- `confidence`: Confidence score [0, 1]

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Model Training

To train the neural note classifier:

1. Prepare a dataset of isolated guitar stems with ground-truth tablature
2. Extract spectrograms and labels
3. Train using PyTorch (training scripts to be added)

## Configuration

Key parameters can be adjusted in the pipeline initialization:

```python
pipeline = TablAIturePipeline(
    target_sr=22050,           # Sample rate
    use_neural_model=False,     # Enable neural classification
    f0_detector=YINF0Detector(), # F0 detection method
    device="cpu"                # Device for neural models
)
```

## Dependencies

- **numpy**: Numerical computations
- **scipy**: Signal processing
- **librosa**: Audio analysis and feature extraction
- **torch**: Deep learning framework
- **soundfile**: Audio I/O


## References

- YIN: A fundamental frequency estimator for speech and music (de Cheveigné & Kawahara, 2002)
- Librosa: Audio and music analysis library
- PyTorch: Deep learning framework


import numpy as np
import whisper
import torch
import string

# This dictionary will act as a simple cache
# so we don't reload the model every time.
_model_cache = {}

def load_model(model_size="base"):
    """
    Loads the Whisper model onto the GPU (if available) and caches it.
    """
    if model_size in _model_cache:
        return _model_cache[model_size]
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model ('{model_size}') onto {device}...")
    model = whisper.load_model(model_size, device=device)
    _model_cache[model_size] = model
    print("Model loaded successfully.")
    return model

def normalize_text(text):
    """
    Normalizes text for a fair WER comparison:
    - Converts to uppercase
    - Removes all punctuation
    - Strips leading/trailing whitespace
    """
    if not text:
        return ""
    return text.upper().translate(str.maketrans('', '', string.punctuation)).strip()

def transcribe_audio(model, audio_data, sample_rate):
    """
    Transcribes audio data (as a NumPy array) using the provided model.
    """
    # Whisper expects audio as float32, normalized between -1.0 and 1.0
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / 32767.0
        
    result = model.transcribe(audio_data, fp16=torch.cuda.is_available())
    return normalize_text(result["text"])


def transcribe_with_timestamps(model, audio_path):
    """
    Transcribes audio and returns the full result dictionary including word-level timestamps.
    Crucial for the GUI search functionality.
    """
    # By passing the file path (string) instead of the numpy array, 
    # Whisper uses ffmpeg to automatically resample the 8kHz NOIZEUS file to 16kHz!
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available(), word_timestamps=True)
    
    return result
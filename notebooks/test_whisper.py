import whisper
import time
import torch

# --- CONFIGURATION ---
MODEL_SIZE = "base"
# Make sure this path points to your clean audio file
AUDIO_FILE = "../data/raw/clean/hello_whisper_test.wav"

# --- SCRIPT ---
# Check if a CUDA-enabled GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(f"Loading Whisper model ('{MODEL_SIZE}')...")
# The model will be downloaded automatically on the first run
model = whisper.load_model(MODEL_SIZE, device=device) 
print("Model loaded successfully.")

print(f"\nStarting transcription of: {AUDIO_FILE}")
start_time = time.time()

# Run the transcription
result = model.transcribe(AUDIO_FILE)

end_time = time.time()
print("Transcription complete.")
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Print the final result
print("\n--- TRANSCRIPTION ---")
print(result["text"].strip())
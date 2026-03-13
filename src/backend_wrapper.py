# File: src/backend_wrapper.py

import os
import numpy as np
import scipy.io.wavfile as wavfile
import string

# Import your team's existing code
from dsp.noise_reduction import spectral_subtraction, wiener_filter, hybrid_filter
from asr.transcribe import load_model, transcribe_with_timestamps

def process_audio_search(input_audio_path, keyword, dsp_method):
    """
    The main pipeline for the GUI.
    1. Loads audio -> 2. Cleans it -> 3. Transcribes it -> 4. Searches for keyword.
    """
    print(f"\n[Backend] Starting process for: {input_audio_path}")
    print(f"[Backend] Selected DSP Method: {dsp_method}")
    
    # --- 1. Load the Audio ---
    sample_rate, noisy_audio = wavfile.read(input_audio_path)
    
    # NEW: Force stereo music to mono by averaging the left and right channels
    if len(noisy_audio.shape) > 1:
        print("[Backend] Converting stereo audio to mono...")
        noisy_audio = np.mean(noisy_audio, axis=1)

    # Normalize if necessary before DSP
    if noisy_audio.dtype != np.float32:
        noisy_audio = noisy_audio.astype(np.float32) / 32767.0

    # --- 2. Apply the chosen DSP Method ---
    print("[Backend] Applying DSP...")
    if dsp_method == "Spectral Subtraction":
        cleaned_audio = spectral_subtraction(noisy_audio, sample_rate)
    elif dsp_method == "Wiener Filter":
        cleaned_audio = wiener_filter(noisy_audio, sample_rate)
    elif dsp_method == "Hybrid Filter":
        cleaned_audio = hybrid_filter(noisy_audio, sample_rate)
    else:
        # "None" or baseline
        cleaned_audio = noisy_audio 

    # Save the cleaned audio to a temporary file so the GUI can play it
    os.makedirs("data/cleaned", exist_ok=True)
    filename = os.path.basename(input_audio_path)
    cleaned_audio_path = f"data/cleaned/GUI_temp_{filename}"
    
    # Convert back to int16 for saving as WAV
    audio_to_save = np.int16(cleaned_audio * 32767)
    wavfile.write(cleaned_audio_path, sample_rate, audio_to_save)

    # --- 3. Transcribe with Whisper ---
    print("[Backend] Running Whisper (this may take a moment)...")
    model = load_model("base") # Or whichever size your team agreed on
    whisper_result = transcribe_with_timestamps(model, cleaned_audio, sample_rate)
    
    # --- 4. Search for the Keyword ---
    print(f"[Backend] Searching for keyword: '{keyword}'")
    matches = []
    target_word = keyword.strip().lower()
    
    # Navigate Whisper's output structure
    for segment in whisper_result.get("segments", []):
        for word_info in segment.get("words", []):
            # Clean punctuation out of Whisper's word guesses
            transcribed_word = word_info["word"].lower().translate(str.maketrans('', '', string.punctuation)).strip()
            
            # If the user's keyword is inside the transcribed word
            if target_word in transcribed_word:
                matches.append({
                    "word": word_info["word"].strip(),
                    "start": round(word_info["start"], 2),
                    "end": round(word_info["end"], 2)
                })

    print(f"[Backend] Found {len(matches)} matches!")
    
    # --- 5. Return data to the GUI ---
    return {
        "status": "success",
        "cleaned_audio_path": cleaned_audio_path,
        "matches_found": len(matches),
        "results": matches
    }

# --- Quick Test ---
# If you run this file directly, it will test itself without the GUI.
if __name__ == "__main__":
    # Change this path to an actual noisy file in your data/raw/ folder to test!
    TEST_FILE = "data/raw/sp01_car_sn5.wav" 
    if os.path.exists(TEST_FILE):
        output = process_audio_search(TEST_FILE, "signal", "Hybrid Filter")
        print("\n--- Final Output Sent to GUI ---")
        print(output)
    else:
        print(f"Please put a test file at {TEST_FILE} to run the standalone test.")
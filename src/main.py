import os
import jiwer
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm  # For a nice progress bar!
import re # Import regex for parsing filenames

# Import our custom modules
from dsp.noise_reduction import spectral_subtraction, wiener_filter, hybrid_filter
from asr.transcribe import load_model, normalize_text, transcribe_audio

# --- 1. EXPERIMENT CONFIGURATION ---

# --- Ground Truth Sentences (from NOIZEUS) ---
GROUND_TRUTH_SENTENCES = [
    "THE BIRCH CANOE SLID ON THE SMOOTH PLANKS",
    "HE KNEW THE SKILL OF THE GREAT YOUNG ACTRESS",
    "HER PURSE WAS FULL OF USELESS TRASH",
    "READ VERSE OUT LOUD FOR PLEASURE",
    "WIPE THE GREASE OFF HIS DIRTY FACE",
    "MEN STRIVE BUT SELDOM GET RICH",
    "WE FIND JOY IN THE SIMPLEST THINGS",
    "HEDGE APPLES MAY STAIN YOUR HANDS GREEN",
    "HURDLE THE PIT WITH THE AID OF A LONG POLE",
    "THE SKY THAT MORNING WAS CLEAR AND BRIGHT BLUE",
    "HE WROTE DOWN A LONG LIST OF ITEMS",
    "THE DRIP OF THE RAIN MADE A PLEASANT SOUND",
    "SMOKE POURED OUT OF EVERY CRACK",
    "HATS ARE WORN TO TEA AND NOT TO DINNER",
    "THE CLOTHES DRIED ON A THIN WOODEN RACK",
    "THE STRAY CAT GAVE BIRTH TO KITTENS",
    "THE LAZY COW LAY IN THE COOL GRASS",
    "THE FRIENDLY GANG LEFT THE DRUG STORE",
    "WE TALKED OF THE SIDESHOW IN THE CIRCUS",
    "THE SET OF CHINA HIT THE FLOOR WITH A CRASH",
    "CLAMS ARE SMALL, ROUND, SOFT AND TASTY",
    "THE LINE WHERE THE EDGES JOIN WAS CLEAN",
    "STOP WHISTLING AND WATCH THE BOYS MARCH",
    "A CRUISE IN WARM WATERS IN A SLEEK YACHT IS FUN",
    "A GOOD BOOK INFORMS OF WHAT WE OUGHT TO KNOW",
    "SHE HAS A SMART WAY OF WEARING CLOTHES",
    "BRING YOUR BEST COMPASS TO THE THIRD CLASS",
    "THE CLUB RENTED THE RINK FOR THE FIFTH NIGHT",
    "THE FLINT SPUTTERED AND LIT A PINE TORCH",
    "LET'S ALL JOIN AS WE SING THE LAST CHORUS"
]

# --- Parameters to Test ---
SS_ALPHAS = [1.0, 2.5, 4.0]  # Over-subtraction
SS_BETAS = [0.01, 0.05]       # Flooring
WF_ALPHAS = [0.95, 0.98]      # Smoothing
HY_ALPHAS = [0.95, 0.98]      # Smoothing

# --- Data to Process ---
NOISE_TYPES = ['car', 'babble']
SNR_LEVELS = ['0dB', '5dB', '15dB']

# --- PATH FIX: Remove ../ from these paths ---
DATA_ROOT = 'data/raw' 
RESULTS_FILE = 'results/experiment_results.csv'
# -------------------------------------------

# --- 2. RUN THE EXPERIMENT ---

def run_experiment():
    # Load the ASR model once
    model = load_model('base')
    
    all_results = []
    
    # Create a list of all files to process
    test_files = []
    for noise in NOISE_TYPES:
        for snr in SNR_LEVELS:
            folder_path = os.path.join(DATA_ROOT, noise, snr)
            if not os.path.isdir(folder_path):
                print(f"Warning: Directory not found, skipping: {folder_path}")
                continue
                
            # --- FILENAME FIX: Read filenames from the directory ---
            for filename in os.listdir(folder_path):
                if filename.endswith('.wav'):
                    # Use regex to find the 'spXX' part
                    match = re.match(r'sp(\d{2})', filename)
                    if match:
                        sentence_index = int(match.group(1)) - 1
                        if 0 <= sentence_index < 30:
                            test_files.append((os.path.join(folder_path, filename), noise, snr, sentence_index))
                        else:
                            print(f"Warning: Could not parse index from {filename}")
    
    # Use tqdm for a progress bar
    for file_path, noise, snr, sentence_index in tqdm(test_files, desc="Running Experiments"):
        try:
            sample_rate, noisy_audio = wavfile.read(file_path)
            
            # Get the correct ground truth text
            ground_truth = normalize_text(GROUND_TRUTH_SENTENCES[sentence_index])
            
            # --- Test 1: Baseline (Noisy Audio) ---
            noisy_transcript = transcribe_audio(model, noisy_audio, sample_rate)
            wer_baseline = jiwer.wer(ground_truth, noisy_transcript)
            all_results.append({
                'file': os.path.basename(file_path), 'noise': noise, 'snr': snr,
                'method': 'Baseline', 'alpha': None, 'beta': None, 'wer': wer_baseline
            })

            # --- Test 2: Spectral Subtraction ---
            for alpha in SS_ALPHAS:
                for beta in SS_BETAS:
                    ss_cleaned = spectral_subtraction(noisy_audio, sample_rate, alpha=alpha, beta=beta)
                    ss_transcript = transcribe_audio(model, ss_cleaned, sample_rate)
                    wer_ss = jiwer.wer(ground_truth, ss_transcript)
                    all_results.append({
                        'file': os.path.basename(file_path), 'noise': noise, 'snr': snr,
                        'method': 'Spectral Subtraction', 'alpha': alpha, 'beta': beta, 'wer': wer_ss
                    })
            
            # --- Test 3: Wiener Filter ---
            for alpha in WF_ALPHAS:
                wf_cleaned = wiener_filter(noisy_audio, sample_rate, alpha=alpha)
                wf_transcript = transcribe_audio(model, wf_cleaned, sample_rate)
                wer_wf = jiwer.wer(ground_truth, wf_transcript)
                all_results.append({
                    'file': os.path.basename(file_path), 'noise': noise, 'snr': snr,
                    'method': 'Wiener Filter', 'alpha': alpha, 'beta': None, 'wer': wer_wf
                })

            # --- Test 4: Hybrid Filter ---
            for alpha in HY_ALPHAS:
                hy_cleaned = hybrid_filter(noisy_audio, sample_rate, alpha=alpha)
                hy_transcript = transcribe_audio(model, hy_cleaned, sample_rate)
                wer_hy = jiwer.wer(ground_truth, hy_transcript)
                all_results.append({
                    'file': os.path.basename(file_path), 'noise': noise, 'snr': snr,
                    'method': 'Hybrid Filter', 'alpha': alpha, 'beta': None, 'wer': wer_hy
                })

        except FileNotFoundError:
            print(f"Warning: File not found {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # --- 3. SAVE RESULTS ---
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_FILE, index=False)
    
    print(f"\nExperiment complete! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_experiment()
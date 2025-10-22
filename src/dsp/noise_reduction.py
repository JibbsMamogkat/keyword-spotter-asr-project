import numpy as np
import scipy.signal
from scipy.io import wavfile

def spectral_subtraction(
    noisy_signal,
    sample_rate,
    noise_duration_s=0.5,
    n_fft=512,
    hop_length=128,
    alpha=2.0,
    beta=0.01
):
    """
    Implements the Spectral Subtraction algorithm for noise reduction.

    This function follows the steps outlined in the project's literature review,
    specifically the methodology for Spectral Subtraction.

    Args:
        noisy_signal (np.array): The input audio signal as a NumPy array.
        sample_rate (int): The sample rate of the audio signal.
        noise_duration_s (float): Duration of the initial noise segment in seconds.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for the STFT.
        alpha (float): The over-subtraction factor.
        beta (float): The spectral flooring factor.

    Returns:
        np.array: The cleaned audio signal.
    """
    
    # --- Step 1: Noise Profile Estimation ---
    # Determine the number of samples for the noise clip
    noise_samples = int(noise_duration_s * sample_rate)
    noise_clip = noisy_signal[:noise_samples]

    # Calculate the STFT of the noise clip
    _, _, noise_stft = scipy.signal.stft(
        noise_clip, fs=sample_rate, nperseg=n_fft, noverlap=hop_length
    )
    
    # Calculate the average noise power spectrum (the "noise profile")
    noise_psd_profile = np.mean(np.abs(noise_stft)**2, axis=1)

    # --- Step 2 & 3: Framing and Subtraction in Frequency Domain ---
    # Calculate the STFT of the entire noisy signal
    _, _, noisy_stft = scipy.signal.stft(
        noisy_signal, fs=sample_rate, nperseg=n_fft, noverlap=hop_length
    )
    
    # Separate magnitude and phase
    noisy_magnitude = np.abs(noisy_stft)
    noisy_phase = np.angle(noisy_stft)
    
    # Calculate the power spectrum of the noisy signal
    noisy_psd = noisy_magnitude**2
    
    # Perform the core spectral subtraction
    # The noise profile needs to be reshaped to be subtracted from each frame
    cleaned_psd = noisy_psd - alpha * noise_psd_profile[:, np.newaxis]

    # --- Step 4: Magnitude Flooring and Artifact Mitigation ---
    # Apply flooring to prevent negative values and reduce "musical noise"
    floor = beta * noise_psd_profile[:, np.newaxis]
    cleaned_psd = np.maximum(cleaned_psd, floor)

    # --- Step 5: Signal Reconstruction ---
    # Get the cleaned magnitude by taking the square root
    cleaned_magnitude = np.sqrt(cleaned_psd)
    
    # Recombine the cleaned magnitude with the original noisy phase
    cleaned_stft = cleaned_magnitude * np.exp(1j * noisy_phase)
    
    # Perform the Inverse STFT to get the cleaned time-domain signal
    _, cleaned_audio = scipy.signal.istft(
        cleaned_stft, fs=sample_rate, nperseg=n_fft, noverlap=hop_length
    )
    
    return cleaned_audio

# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    # This block will only run when you execute this script directly
    
    # --- Configuration ---
    INPUT_FILE = 'data/raw/car/5dB/sp01_car_sn5.wav' # Example noisy file
    OUTPUT_FILE = 'data/cleaned/sp01_car_sn5_cleaned.wav'
    
    print(f"Loading audio from: {INPUT_FILE}")
    sample_rate, noisy_audio = wavfile.read(INPUT_FILE)
    
    # Ensure audio is float for processing
    if noisy_audio.dtype != np.float32:
        noisy_audio = noisy_audio / 32767.0
        
    print("Applying Spectral Subtraction...")
    cleaned_audio = spectral_subtraction(
        noisy_signal=noisy_audio,
        sample_rate=sample_rate,
        alpha=2.0, # Experiment with this value
        beta=0.01  # Experiment with this value
    )
    
    print(f"Saving cleaned audio to: {OUTPUT_FILE}")
    # Convert back to 16-bit integer for standard wav format
    cleaned_audio_int = np.int16(cleaned_audio * 32767)
    wavfile.write(OUTPUT_FILE, sample_rate, cleaned_audio_int)
    
    print("Done.")
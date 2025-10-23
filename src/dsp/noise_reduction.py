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

def wiener_filter(
    noisy_signal,
    sample_rate,
    noise_duration_s=0.5,
    n_fft=512,
    hop_length=128,
    alpha=0.98
):
    """
    Implements a Wiener filter for noise reduction using recursive noise estimation.

    This function is based on the methodology from Upadhyay et al. (2016).

    Args:
        noisy_signal (np.array): The input audio signal.
        sample_rate (int): The sample rate of the audio.
        noise_duration_s (float): Duration of the initial noise segment.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for the STFT.
        alpha (float): The smoothing parameter for recursive noise estimation.

    Returns:
        np.array: The cleaned audio signal.
    """
    
    # --- Step 1: Signal Framing and STFT ---
    _, _, noisy_stft = scipy.signal.stft(
        noisy_signal, fs=sample_rate, nperseg=n_fft, noverlap=hop_length
    )
    noisy_psd = np.abs(noisy_stft)**2

    # --- Step 2: Noise Power Estimation (Recursive Averaging) ---
    # Initialize noise estimate from the first few frames
    noise_samples = int(noise_duration_s * sample_rate)
    num_noise_frames = int(np.ceil(noise_samples / hop_length))
    
    initial_noise_psd = np.mean(noisy_psd[:, :num_noise_frames], axis=1)
    
    # Initialize the smoothed noise PSD array
    smoothed_noise_psd = np.zeros_like(noisy_psd)
    smoothed_noise_psd[:, 0] = initial_noise_psd
    
    # Loop through frames to apply recursive averaging
    for k in range(1, noisy_psd.shape[1]):
        smoothed_noise_psd[:, k] = alpha * smoothed_noise_psd[:, k-1] + (1 - alpha) * noisy_psd[:, k]

    # --- Step 3: A Priori Clean Signal Power Estimation ---
    # Estimate the clean signal's power by subtracting the smoothed noise estimate
    signal_psd_estimate = np.maximum(noisy_psd - smoothed_noise_psd, 0)

    # --- Step 4: Wiener Filter Construction ---
    # The filter is the ratio of estimated signal power to estimated total power
    # Adding a small epsilon to avoid division by zero
    epsilon = 1e-10
    wiener_gain = signal_psd_estimate / (signal_psd_estimate + smoothed_noise_psd + epsilon)
    
    # --- Step 5: Signal Reconstruction ---
    # Apply the filter to the complex STFT of the noisy signal
    cleaned_stft = noisy_stft * wiener_gain
    
    # Perform the Inverse STFT
    _, cleaned_audio = scipy.signal.istft(
        cleaned_stft, fs=sample_rate, nperseg=n_fft, noverlap=hop_length
    )

    return cleaned_audio

def hybrid_filter(
    noisy_signal,
    sample_rate,
    noise_duration_s=0.5,
    n_fft=512,
    hop_length=128,
    alpha=0.98
):
    """
    Implements a hybrid Wiener-Spectral Subtraction filter.

    This method is inspired by Pardede et al. (2019), using a Wiener
    filter for instantaneous noise estimation within a recursive averaging
    framework for a final spectral subtraction step.
    
    Args:
        noisy_signal (np.array): The input audio signal.
        sample_rate (int): The sample rate of the audio.
        noise_duration_s (float): Duration of the initial noise segment.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for the STFT.
        alpha (float): The smoothing parameter for recursive noise estimation.

    Returns:
        np.array: The cleaned audio signal.
    """
    # --- Step 1: Initialization ---
    _, _, noisy_stft = scipy.signal.stft(
        noisy_signal, fs=sample_rate, nperseg=n_fft, noverlap=hop_length
    )
    noisy_psd = np.abs(noisy_stft)**2
    
    # Initialize noise and speech power estimates from the first few frames
    noise_samples = int(noise_duration_s * sample_rate)
    num_noise_frames = int(np.ceil(noise_samples / hop_length))
    
    # Initial noise estimate
    noise_psd_est = np.mean(noisy_psd[:, :num_noise_frames], axis=1)
    # Initial speech estimate (can be zero or a small value)
    speech_psd_est = np.zeros_like(noise_psd_est)

    # Prepare array for final cleaned PSD
    cleaned_psd = np.zeros_like(noisy_psd)
    
    # --- Frame-by-frame processing loop ---
    for k in range(noisy_psd.shape[1]):
        current_noisy_psd = noisy_psd[:, k]

        # --- Step 2: Instantaneous Noise Estimation via Wiener Filter ---
        # Construct a noise-passing Wiener filter from PREVIOUS frame's estimates
        epsilon = 1e-10
        noise_wiener_gain = noise_psd_est / (speech_psd_est + noise_psd_est + epsilon)
        
        # Apply it to the CURRENT frame's power to get an instant noise estimate
        instant_noise_psd = noise_wiener_gain * current_noisy_psd
        
        # --- Step 3: Noise Profile Smoothing (Recursive Averaging) ---
        # Update the running noise estimate using the new instant estimate
        noise_psd_est = alpha * noise_psd_est + (1 - alpha) * instant_noise_psd
        
        # --- Step 4: Final Enhancement via Spectral Subtraction ---
        # Perform spectral subtraction with the NEW, refined noise estimate
        current_cleaned_psd = np.maximum(current_noisy_psd - noise_psd_est, 0)
        cleaned_psd[:, k] = current_cleaned_psd
        
        # --- Step 6: Update for Next Frame ---
        # The cleaned speech power from this frame becomes the estimate for the next
        speech_psd_est = current_cleaned_psd

    # --- Step 5: Signal Reconstruction ---
    cleaned_magnitude = np.sqrt(cleaned_psd)
    cleaned_stft = cleaned_magnitude * np.exp(1j * np.angle(noisy_stft))
    
    _, cleaned_audio = scipy.signal.istft(
        cleaned_stft, fs=sample_rate, nperseg=n_fft, noverlap=hop_length
    )

    return cleaned_audio

# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    # This block will only run when you execute this script directly
    
    # --- Configuration ---
    INPUT_FILE = 'data/raw/car/5dB/sp01_car_sn5.wav' # Example noisy file
    OUTPUT_FILE_SS = 'data/cleaned/sp01_ss_cleaned.wav'
    OUTPUT_FILE_WF = 'data/cleaned/sp01_wf_cleaned.wav'
    OUTPUT_FILE_HY = 'data/cleaned/sp01_hy_cleaned.wav'
    
    print(f"Loading audio from: {INPUT_FILE}")
    sample_rate, noisy_audio = wavfile.read(INPUT_FILE)
    
    if noisy_audio.dtype != np.float32:
        noisy_audio = noisy_audio / 32767.0
        
    # --- Test Spectral Subtraction ---
    print("Applying Spectral Subtraction...")
    ss_cleaned = spectral_subtraction(noisy_audio, sample_rate)
    ss_cleaned_int = np.int16(ss_cleaned * 32767)
    wavfile.write(OUTPUT_FILE_SS, sample_rate, ss_cleaned_int)
    print(f"SS output saved to: {OUTPUT_FILE_SS}")
    
    # --- Test Wiener Filter ---
    print("\nApplying Wiener Filter...")
    wf_cleaned = wiener_filter(noisy_audio, sample_rate)
    wf_cleaned_int = np.int16(wf_cleaned * 32767)
    wavfile.write(OUTPUT_FILE_WF, sample_rate, wf_cleaned_int)
    print(f"WF output saved to: {OUTPUT_FILE_WF}")

    # --- Test Hybrid Filter ---
    print("\nApplying Hybrid Filter...")
    hy_cleaned = hybrid_filter(noisy_audio, sample_rate)
    wavfile.write(OUTPUT_FILE_HY, sample_rate, np.int16(hy_cleaned * 32767))
    print(f"HY output saved to: {OUTPUT_FILE_HY}")
    
    print("\nDone.")
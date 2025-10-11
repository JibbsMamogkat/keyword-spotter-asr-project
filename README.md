# DSP-Enhanced Keyword Spotting System

**Status:** In Development

This project is a Python-based application designed to find spoken keywords in noisy audio files. It features custom-built Digital Signal Processing (DSP) modules for audio cleaning and leverages a self-hosted, state-of-the-art Automatic Speech Recognition (ASR) model (OpenAI's Whisper) for transcription.

This project investigates the critical trade-off between noise reduction and the introduction of processing artifacts that can degrade ASR performance. Our primary goal is to implement, compare, and tune classic DSP noise reduction algorithms to **maximize the final ASR accuracy**, not just subjective listening quality.

This project is being developed for the CPE113L-1 Digital Signal Processing (Laboratory) course.

---

## 🚀 Key Features

* **DSP Noise Reduction:** Implements and compares classic DSP noise reduction algorithms (**Spectral Subtraction** and **Wiener Filtering**) to clean audio signals before recognition.
* **Self-Hosted ASR:** Uses OpenAI's Whisper model locally for fast, private, and powerful speech-to-text transcription.
* **Open-Vocabulary Search:** Allows the user to search for any typed keyword, not just a pre-defined set.
* **Performance Analysis:** Designed to empirically measure and compare the ASR accuracy improvement provided by each DSP cleaning module.

---

## ⚙️ System Workflow

The system operates as a sequential pipeline:

`[Noisy Audio File]` → `[Our DSP Cleaning Module (SS or Wiener)]` → `[Cleaned Audio]` → `[Whisper ASR Model]` → `[Timestamped Transcript]` → `[Text Search]` → `[Final Timestamps]`

---

## 📂 Folder Structure

```
keyword-asr-project/
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/            # Input noisy audio files
│   └── cleaned/        # Output of the DSP module
│
├── notebooks/          # Jupyter notebooks for research & prototyping
│
├── results/            # Stores output transcripts and plots
│   ├── plots/
│   └── transcripts/
│
├── src/                # All source code
│   ├── dsp/            # DSP cleaning module
│   │   └── noise_reduction.py
│   ├── asr/            # Wrapper for the Whisper ASR model
│   │   └── transcribe.py
│   └── main.py         # Main script to run the pipeline
│
└── models/             # Not used for Whisper, but good practice
```

---

## 🛠️ Setup and Installation

Follow these steps to set up the development environment.

**1. Prerequisites:**
* Python 3.8+
* Git

**2. Clone the Repository:**
```bash
git clone <your-repo-url>
cd keyword-asr-project
```

**3. Set Up a Python Virtual Environment:**
```bash
python -m venv venv
# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**4. Install PyTorch with CUDA Support:**
This is the most critical step to enable GPU acceleration. Go to the [Official PyTorch Website](https://pytorch.org/get-started/locally/) and select the correct options (e.g., Stable, Windows/Linux, Pip, Python, CUDA) to get your specific installation command.

**5. Install FFmpeg:**
Whisper requires FFmpeg to be installed on your system.
* **Windows:** Use Chocolatey: `choco install ffmpeg`
* **macOS:** Use Homebrew: `brew install ffmpeg`
* **Linux:** Use apt: `sudo apt update && sudo apt install ffmpeg`

**6. Install Python Dependencies:**
Once PyTorch and FFmpeg are set up, install the rest of the libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Use

The main script is designed to be run from the command line, with an option to select the cleaning method.

```bash
python src/main.py --audio_file "path/to/your/audio.wav" --keyword "your_keyword" --method "spectral_subtraction"
```
**Example:**
```bash
python src/main.py --audio_file "data/raw/noisy_lecture.wav" --keyword "Fourier" --method "wiener"
```

---

## 📝 4-Week Development Plan & To-Do List

This is our high-level plan to guide the development process.

### **Week 1: Foundations & Setup**
- [ ] Finalize project scope and choose target noisy audio for testing. *(Assigned to: All)*
- [ ] Set up the complete ASR environment (PyTorch+CUDA, FFmpeg, Whisper). *(Assigned to: ____)*
- [ ] **Milestone:** Successfully transcribe a clean audio file using a "Hello Whisper" script. *(Assigned to: ____)*
- [ ] Research and understand **Spectral Subtraction** and **Wiener Filtering** algorithms. *(Assigned to: ____)*

### **Week 2: DSP Module Implementation**
- [ ] Write the Python code for the **Spectral Subtraction** algorithm in `src/dsp/noise_reduction.py`. *(Assigned to: ____)*
- [ ] Implement a simplified **Wiener Filtering** algorithm in the same module. *(Assigned to: ____)*
- [ ] Create a test notebook in `notebooks/` to test both cleaning functions. *(Assigned to: ____)*
- [ ] **Milestone:** Successfully clean a noisy audio file using **both methods** and verify the results by listening and viewing spectrograms. *(Assigned to: ____)*

### **Week 3: Integration & Experimentation**
- [ ] Write the ASR wrapper in `src/asr/transcribe.py` to handle Whisper transcription and timestamp extraction. *(Assigned to: ____)*
- [ ] Write the `main.py` script to connect the DSP module and ASR module, allowing method selection. *(Assigned to: ____)*
- [ ] **Milestone:** Run the full pipeline on a noisy audio file and successfully get a timestamped transcript. *(Assigned to: ____)*
- [ ] **Design and run the final experiment: compare ASR accuracy on 1) the original noisy audio, 2) audio cleaned with Spectral Subtraction, and 3) audio cleaned with Wiener Filtering.** *(Assigned to: All)*

### **Week 4: Finalization**
- [ ] Analyze the experiment results and create plots/tables, focusing on comparing the effectiveness of the two DSP methods. *(Assigned to: ____)*
- [ ] Write the final project report. *(Assigned to: All)*
- [ ] Create the final presentation slides. *(Assigned to: All)*
- [ ] **Milestone:** Final project submission and presentation. *(Assigned to: All)*

---

## 👥 Team Members

* Duff Bastasa
* Mohammad Jameel Jibreel Mamogkat
* Xavier Fuentes
* Ronnie Borromeo
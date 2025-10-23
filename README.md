# DSP-Enhanced Keyword Spotting System

**Status:** In Development

This project is a Python-based application designed to find spoken keywords in noisy audio files. It features custom-built Digital Signal Processing (DSP) modules for audio cleaning and leverages a self-hosted, state-of-the-art Automatic Speech Recognition (ASR) model (OpenAI's Whisper) for transcription.

The core of this project is a practical investigation into the central challenge of speech enhancement for ASR: the trade-off between noise reduction and the introduction of processing artifacts that can degrade ASR performance. Our primary goal is to implement, compare, and tune classic DSP noise reduction algorithms to **maximize the final ASR accuracy**, not just subjective listening quality.

This project is being developed for the CPE113L-1 Digital Signal Processing (Laboratory) course.

---

## 🚀 Key Features

* **Comparative DSP Implementation:** Implements and compares classic DSP noise reduction algorithms, including **Spectral Subtraction** and **Wiener Filtering**.
* **Advanced Hybrid Method:** Explores an advanced hybrid technique inspired by recent literature that combines principles from both methods to better handle non-stationary noise.
* **Self-Hosted ASR:** Uses OpenAI's Whisper model locally for fast, private, and powerful speech-to-text transcription.
* **Performance Analysis:** Designed to empirically measure the ASR accuracy improvement provided by each DSP cleaning module using the Word Error Rate (WER) metric.

---

## ⚙️ System Workflow

The system operates as a sequential pipeline:

`[Noisy Audio File]` → `[Our DSP Cleaning Module (SS, WF, or Hybrid)]` → `[Cleaned Audio]` → `[Whisper ASR Model]` → `[Timestamped Transcript]` → `[Text Search]` → `[Final Timestamps]`

---

## 📂 Folder Structure

```
keyword-asr-project/
├── .gitignore
├── README.md
├── dsp_lab_lit_review.pdf
├── requirements.txt
│
├── data/
│   ├── raw/            # Input noisy audio files (e.g., from NOIZEUS)
│   └── cleaned/        # Output of the DSP modules
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
│   └── main.py         # Main script to run the automated experiment
│
└── venv/               # Python virtual environment
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

The main script is designed to run the full automated experiment and save the results.

```bash
# To run the full experiment on all configured data and methods
python src/main.py
```

Individual methods can also be tested from within the `src/dsp/noise_reduction.py` script for quick verification.

---

## 📝 4-Week Development Plan & To-Do List

This is our high-level plan to guide the development process.

### **Week 1: Foundations & Setup (Complete)**
- [x] Finalize project scope and choose target noisy audio for testing.
- [x] Set up the complete ASR environment (PyTorch+CUDA, FFmpeg, Whisper).
- [x] **Milestone:** Successfully transcribe a clean audio file using a "Hello Whisper" script.
- [x] Research and understand **Spectral Subtraction** and **Wiener Filtering** algorithms.

### **Week 2: DSP Module Implementation**
- [x] Implement the **Spectral Subtraction** algorithm in `src/dsp/noise_reduction.py`. *(Assigned to: ____)*
- [x] Implement the **Wiener Filtering** algorithm in the same module. *(Assigned to: ____)*
- [x] (Optional) Implement the **Hybrid Method** inspired by the literature. *(Assigned to: ____)*
- [x] Create a test notebook in `notebooks/` to test all implemented cleaning functions individually. *(Assigned to: ____)*
- [x] **Milestone:** Successfully clean a noisy audio file using **all implemented methods** and verify the results by listening and viewing spectrograms. *(Assigned to: ____)*

### **Week 3: Integration & Experimentation**
- [ ] Write the ASR wrapper in `src/asr/transcribe.py` to handle Whisper transcription and timestamp extraction. *(Assigned to: ____)*
- [ ] Write the `main.py` script to automate the experiment. It should loop through all test files and all DSP methods. *(Assigned to: ____)*
- [ ] **Milestone:** Run the full automated pipeline and generate the final `results.csv` file. *(Assigned to: All)*
- [ ] Begin analyzing the results and generating initial plots. *(Assigned to: All)*

### **Week 4: Finalization**
- [ ] Complete the analysis of the experiment results, creating plots and tables that compare the effectiveness of the DSP methods. *(Assigned to: ____)*
- [ ] Write the final project report, focusing on the methodology, results, and discussion of the trade-offs discovered. *(Assigned to: All)*
- [ ] Create the final presentation slides. *(Assigned to: All)*
- [ ] **Milestone:** Final project submission and presentation. *(Assigned to: All)*

---

## 👥 Team Members

* Duff Bastasa
* Mohammad Jameel Jibreel Mamogkat
* Xavier Fuentes
* Ronnie Borromeo
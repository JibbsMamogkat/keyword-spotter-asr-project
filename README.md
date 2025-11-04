# DSP-Enhanced Audio Search ("Ctrl+F" for Audio)

**Status:** Phase 4 (ML Development) In Progress

This project is a Python-based application designed to find spoken keywords in noisy audio files. It functions as a "Ctrl+F for audio," allowing a user to load a noisy audio file, type a keyword, and receive a list of timestamps where that word was spoken.

To achieve this, the system first cleans the audio using an advanced enhancement front-end. The core of this project is a comprehensive investigation to find the best enhancement method:
1.  **Classic DSP Methods:** We implement and compare Spectral Subtraction, Wiener Filtering, and a Hybrid method.
2.  **Modern ML Method:** We build a deep learning (CNN) model as a state-of-the-art benchmark.
3.  **ASR Backend:** We leverage a self-hosted Whisper ASR model to perform the transcription.

The final, objective metric for success is the **Word Error Rate (WER)**. Our goal is to find the enhancement method that provides the lowest WER, proving its effectiveness for machine listening.

This project is being developed for the CPE113L-1 Digital Signal Processing (Laboratory) course.

---

## 🚀 Key Features

* **Comparative DSP Implementation:** Implements and compares classic DSP noise reduction algorithms: **Spectral Subtraction**, **Wiener Filtering**, and an advanced **Hybrid Method**.
* **State-of-the-Art ML Implementation:** Includes a **Machine Learning (CNN)**-based audio denoiser to be compared against the classic DSP methods.
* **Self-Hosted ASR:** Uses OpenAI's Whisper model (GPU-accelerated) for fast, private, and powerful speech-to-text transcription with word-level timestamps.
* **"Ctrl+F" Functionality:** The final GUI application allows a user to load an audio file, type a keyword, and instantly find all occurrences (with timestamps) of that word in the file.
* **Quantitative Performance Analysis:** All enhancement methods are empirically measured and compared using the Word Error Rate (WER) metric to find the optimal solution.

---

## ⚙️ System Workflow

The final application will operate on the following workflow:

`[User Loads Noisy Audio File]` → `[User Types Keyword]` → `[Python GUI (Tkinter/PyQt)]` → `[Best Enhancement Module (DSP or ML)]` → `[Cleaned Audio]` → `[Whisper ASR (with Timestamps)]` → `[Text Search Logic]` → `[Display List of Timestamps]`

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
│   ├── cleaned/        # Output of the DSP modules
│   └── training/       # (New) For generated (noisy, clean) ML training pairs
│
├── models/             # (New) To store the trained ML model (e.g., cnn_denoiser.h5)
│
├── notebooks/          # Jupyter notebooks for research & prototyping
│
├── results/            # Stores output transcripts and plots
│
├── src/                # All source code
│   ├── dsp/            # DSP cleaning module
│   │   └── noise_reduction.py
│   ├── asr/            # Wrapper for the Whisper ASR model
│   │   └── transcribe.py
│   ├── ml/             # (New) ML model definition and training scripts
│   │   ├── build_dataset.py
│   │   └── train.py
│   ├── main.py         # Main script to run the automated experiment
│   └── app.py          # (New) The final GUI application
│
└── venv/               # Python virtual environment
```

---

## 🛠️ Setup and Installation

*(No changes from your previous version)*

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
Go to the [Official PyTorch Website](https://pytorch.org/get-started/locally/) and run their recommended install command for your system (Pip, CUDA).

**5. Install FFmpeg:**
* **Windows:** `choco install ffmpeg`
* **macOS:** `brew install ffmpeg`
* **Linux:** `sudo apt update && sudo apt install ffmpeg`

**6. Install Python Dependencies:**
```bash
pip install -r requirements.txt
```

---

## 📝 Extended Project Plan & To-Do List

This is our extended plan, which includes our completed experimental phase and the new development phases.

### **Phase 1: Foundations & Setup (Weeks 1) - Complete**
- [x] Finalized project scope and chose the NOIZEUS corpus for testing.
- [x] Set up the complete ASR environment (PyTorch+CUDA, FFmpeg, Whisper).
- [x] **Milestone:** Successfully transcribed a clean audio file using the GPU.
- [x] Researched **Spectral Subtraction** and **Wiener Filtering** algorithms.

### **Phase 2: DSP Module Implementation (Weeks 2) - Complete**
- [x] Implemented **Spectral Subtraction**, **Wiener Filtering**, and the **Hybrid Method** in `src/dsp/noise_reduction.py`.
- [x] **Milestone:** Verified that all three DSP methods successfully process noisy audio and produce cleaned output files.

### **Phase 3: Integration & DSP Experimentation (Weeks 3) - Complete**
- [x] Wrote the ASR wrapper (`src/asr/transcribe.py`) and the automated experiment script (`src/main.py`).
- [x] **Milestone:** Ran the full automated pipeline and generated the final `results.csv` file, comparing all DSP methods against the baseline.
- [x] **Key Finding:** Analysis of `results.csv` shows that while classic DSP methods provide improvement in high-noise (0-5dB), they are harmful in low-noise (15dB), justifying the need for a more advanced approach.

---
### **Phase 4: Machine Learning (ML) Enhancement (Weeks 4-5)**
- [ ] **Research:** Learn and decide on a simple CNN/U-Net architecture for audio denoising (based on our literature).
- [ ] **Data Preparation:** Write a script (`src/ml/build_dataset.py`) to create a training dataset of `(noisy_spectrogram, clean_spectrogram)` pairs from our source files.
- [ ] **Implementation & Training:** Build the ML model in Keras/TensorFlow and train it on the new dataset in a notebook (`notebooks/ml_model.ipynb`).
- [ ] **Milestone:** Save a trained ML model (`models/cnn_denoiser.h5`) that can clean a noisy spectrogram.

### **Phase 5: Final "Bake-Off" Experiment (Week 6)**
- [ ] **Integration:** Add a new `ml_enhancer()` function to `src/dsp/noise_reduction.py` that loads the trained model and runs inference.
- [ ] **Modify `src/main.py`:** Add the new `ml_enhancer()` function to the main experiment loop.
- [ ] **Milestone:** Run the full automated experiment one last time, now including the ML model.
- [ ] **Analysis:** Update `analyze_results.ipynb` to plot all methods (Baseline, SS, WF, Hybrid, ML) and determine the single best-performing method.

### **Phase 6: GUI "Ctrl+F" Application (Weeks 7-8)**
- [ ] **GUI Design:** Choose a framework (e.g., Tkinter) and build the user interface for `src/app.py`. (Buttons: Load Audio, Search. Fields: Keyword, Results).
- [ ] **Backend Logic:** Implement the "search" button functionality:
    1.  Load the audio file.
    2.  Clean the audio using the **best-performing method** from Phase 5.
    3.  Run Whisper transcription on the cleaned audio with `word_timestamps=True`.
    4.  Implement the text search logic to find the keyword and extract its timestamps.
    5.  Display the list of timestamps in the GUI.
- [ ] **Milestone:** A functional, user-friendly application that can successfully find a typed keyword in a noisy audio file.

### **Phase 7: Final Documentation (Week 9)**
- [ ] Write the final project report, focusing on the comparison between classic DSP and modern ML methods.
- [ ] Create the final presentation slides, including a video demo of the GUI application.
- [ ] **Milestone:** Final project submission and presentation.

---

## 👥 Team Members

* Duff Bastasa
* Mohammad Jameel Jibreel Mamogkat
* Xavier Fuentes
* Ronnie Borromeo
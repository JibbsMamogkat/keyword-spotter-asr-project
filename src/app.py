import streamlit as st
import os
import time


from backend_wrapper import process_audio_search
# --- SETUP & CONFIGURATION ---
# This sets the browser tab title and layout width
st.set_page_config(page_title="Audio Ctrl+F", layout="wide")

# Ensure a temporary directory exists for saved uploads
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- 1. BUILD BASIC LAYOUT (Purple Boxes) ---
st.title("🎙️ DSP-Enhanced Audio Search ('Ctrl+F' for Audio)")
st.markdown("Find spoken keywords in noisy audio files using classic DSP and Whisper ASR.")

# We can use a sidebar for inputs to keep the main area clean for results
st.sidebar.header("Input Settings")

# File Uploader
uploaded_file = st.sidebar.file_uploader("1. Upload Noisy Audio (.wav)", type=["wav"])

# Algorithm Dropdown
dsp_method = st.sidebar.selectbox(
    "2. Select Enhancement Method",
    ("Spectral Subtraction", "Wiener Filter", "Hybrid Filter", "Bypass (No DSP)")
)

# Search Box
keyword = st.sidebar.text_input("3. Keyword to Find", placeholder="e.g., hello")

# --- 2. 'PROCESS' BUTTON CLICKED (Purple Box) ---
process_button = st.sidebar.button("Process & Search", type="primary")

if process_button:
    # --- DECISION: IS A FILE LOADED? (Blue Diamond) ---
    if uploaded_file is None:
        # False path (Purple Box)
        st.error("🚨 Please upload a .wav file first!")
    elif not keyword.strip():
        # Extra safety check for the keyword
        st.error("🚨 Please enter a keyword to search for!")
    else:
        # True path (Blue Box)
        
        # Save Uploaded File to Temp Folder
        temp_file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File '{uploaded_file.name}' loaded successfully!")

        # --- THE REAL PROCESSING (Calling Member 2's Code) ---
        with st.spinner(f"Running {dsp_method} and transcribing with Whisper..."):
            
            try:
                # Call the backend pipeline
                backend_response = process_audio_search(
                    input_audio_path=temp_file_path, 
                    keyword=keyword, 
                    dsp_method=dsp_method
                )
                
                # Extract the data from Member 2's dictionary
                cleaned_audio_path = backend_response["cleaned_audio_path"]
                found_timestamps = backend_response["results"]
                
            except Exception as e:
                # If Whisper or the DSP crashes, this stops the whole app from turning red
                st.error(f"An error occurred in the backend processing: {e}")
                st.stop()
            
            # --- 3. UPDATE UI (Results Dashboard) ---
            st.divider()
            st.subheader("📊 Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**1. Original Noisy Audio:**")
                st.audio(temp_file_path) # Plays the raw uploaded file
                
            with col2:
                st.markdown(f"**2. Cleaned Audio ({dsp_method}):**")
                st.audio(cleaned_audio_path) # Plays the processed file from Member 2!
            
            st.divider()

            # --- NEW: Show the full transcript ---
            full_transcript = backend_response.get("full_text", "No text found.")
            st.markdown(f"**📝 Full Whisper Transcript:** _{full_transcript}_")
            st.divider()
            
            # Display List of Timestamps
            if found_timestamps:
                st.write(f"Found **{len(found_timestamps)}** occurrences of '{keyword}':")
                
                for match in found_timestamps:
                    st.info(f"⏱️ **[{match['start']}s - {match['end']}s]** -> {match['word']}")
            else:
                st.warning(f"Keyword '{keyword}' not found in the audio.")
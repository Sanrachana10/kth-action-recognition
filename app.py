import streamlit as st
import keras
import numpy as np
import cv2
import tempfile
import os
from huggingface_hub import hf_hub_download

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="KTH Action Recognition", page_icon="🎬", layout="centered")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Sanrachana/kth-action-model",
        filename="KTH_Final_Model.keras"
    )
    model = keras.saving.load_model(model_path)
    return model

model = load_model()

# --- 3. CONSTANTS ---
ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
MAX_FRAMES = 15
SIZE = (64, 64)

# --- 4. VIDEO PROCESSING ---
def process_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_interval = max(1, total_frames // MAX_FRAMES)

    for i in range(MAX_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, SIZE)
        frames.append(frame / 255.0)

    cap.release()

    while len(frames) < MAX_FRAMES:
        frames.append(np.zeros(SIZE))

    return np.array(frames)

# --- 5. UI ---
st.title("🎬 KTH Action Recognition")
st.markdown("Upload a short video and the AI will classify the human action in it.")
st.divider()

st.markdown("**Supported Actions:** Boxing · Handclapping · Handwaving · Jogging · Running · Walking")
st.divider()

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.video(uploaded_file)

    with st.spinner("Analyzing video..."):
        frames = process_video(tmp_path)
        frames = frames.reshape(1, MAX_FRAMES, SIZE[0], SIZE[1], 1)
        predictions = model.predict(frames, verbose=0)
        predicted_idx = np.argmax(predictions)
        predicted_action = ACTIONS[predicted_idx]
        confidence = float(np.max(predictions)) * 100

    os.unlink(tmp_path)

    # --- 6. RESULTS ---
    st.divider()
    st.subheader("Results")
    st.success(f"Predicted Action: **{predicted_action.upper()}** ({confidence:.1f}% confidence)")

    st.markdown("**Confidence per action:**")
    for i, action in enumerate(ACTIONS):
        st.progress(float(predictions[0][i]), text=f"{action.capitalize()}: {float(predictions[0][i])*100:.1f}%")

st.divider()
st.caption("Developed for Pre-Final Year Lab Assignment - 2026")

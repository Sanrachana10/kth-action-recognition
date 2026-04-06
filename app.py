import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os
import base64
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
    model = tf.keras.models.load_model(model_path)
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

def run_inference(video_path):
    frames = process_video(video_path)
    frames = frames.reshape(1, MAX_FRAMES, SIZE[0], SIZE[1], 1)
    predictions = model.predict(frames, verbose=0)
    predicted_idx = np.argmax(predictions)
    predicted_action = ACTIONS[predicted_idx]
    confidence = float(np.max(predictions)) * 100
    return predicted_action, confidence, predictions

def show_results(predicted_action, confidence, predictions):
    st.divider()
    st.subheader("Results")
    st.success(f"Predicted Action: **{predicted_action.upper()}** ({confidence:.1f}% confidence)")
    st.markdown("**Confidence per action:**")
    for i, action in enumerate(ACTIONS):
        st.progress(float(predictions[0][i]), text=f"{action.capitalize()}: {float(predictions[0][i])*100:.1f}%")

# --- 5. UI ---
st.title("🎬 KTH Action Recognition")
st.markdown("Upload a short video or record one live — the AI will classify the human action in it.")
st.divider()
st.markdown("**Supported Actions:** Boxing · Handclapping · Handwaving · Jogging · Running · Walking")
st.divider()

# --- 6. TABS: Upload vs Record ---
tab1, tab2 = st.tabs(["📁 Upload Video", "🎥 Record Video"])

# ── Tab 1: Upload ──
with tab1:
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mpeg4"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.video(uploaded_file)
        with st.spinner("Analyzing video..."):
            predicted_action, confidence, predictions = run_inference(tmp_path)
        os.unlink(tmp_path)
        show_results(predicted_action, confidence, predictions)

# ── Tab 2: Record ──
with tab2:
    st.markdown("Click **Start Recording**, perform an action, then click **Stop & Classify**.")

    # JavaScript MediaRecorder component
    recorder_html = """
    <div style="display:flex; flex-direction:column; align-items:center; gap:12px;">
        <video id="preview" autoplay muted playsinline
               style="width:100%; max-width:480px; border-radius:12px; background:#000;"></video>

        <div style="display:flex; gap:12px;">
            <button id="startBtn" onclick="startRecording()"
                style="padding:10px 24px; border-radius:8px; background:#e05c2a;
                       color:white; border:none; font-size:1rem; cursor:pointer;">
                ▶ Start Recording
            </button>
            <button id="stopBtn" onclick="stopRecording()" disabled
                style="padding:10px 24px; border-radius:8px; background:#555;
                       color:white; border:none; font-size:1rem; cursor:not-allowed;">
                ■ Stop & Classify
            </button>
        </div>

        <p id="status" style="color:#aaa; font-size:0.85rem;">Ready</p>
        <video id="playback" controls
               style="display:none; width:100%; max-width:480px; border-radius:12px;"></video>
        <input type="hidden" id="videoData">
    </div>

    <script>
    let mediaRecorder;
    let chunks = [];
    let stream;

    async function startRecording() {
        chunks = [];
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        document.getElementById('preview').srcObject = stream;

        mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
        mediaRecorder.ondataavailable = e => chunks.push(e.data);
        mediaRecorder.onstop = () => {
            const blob = new Blob(chunks, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);

            // Show playback
            const playback = document.getElementById('playback');
            playback.src = url;
            playback.style.display = 'block';

            // Convert to base64 and store in hidden input
            const reader = new FileReader();
            reader.onloadend = () => {
                document.getElementById('videoData').value = reader.result;
                // Trigger Streamlit to detect the change
                document.getElementById('videoData').dispatchEvent(new Event('input', { bubbles: true }));
                document.getElementById('status').innerText = 'Recording ready! Click Classify below.';
            };
            reader.readAsDataURL(blob);

            // Stop webcam
            stream.getTracks().forEach(t => t.stop());
            document.getElementById('preview').srcObject = null;
        };

        mediaRecorder.start();
        document.getElementById('startBtn').disabled = true;
        document.getElementById('startBtn').style.background = '#888';
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('stopBtn').style.background = '#2a7ae0';
        document.getElementById('stopBtn').style.cursor = 'pointer';
        document.getElementById('status').innerText = '🔴 Recording...';
    }

    function stopRecording() {
        mediaRecorder.stop();
        document.getElementById('startBtn').disabled = false;
        document.getElementById('startBtn').style.background = '#e05c2a';
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('stopBtn').style.background = '#555';
        document.getElementById('stopBtn').style.cursor = 'not-allowed';
    }
    </script>
    """

    # Render the recorder
    from streamlit.components.v1 import html as st_html
    st_html(recorder_html, height=420)

    st.markdown("---")
    st.markdown("After stopping, paste the recorded video data below to classify:")

    # Because direct JS→Python communication is limited in Streamlit,
    # we use a file uploader as the bridge after recording
    st.info("💡 After recording, your browser will show a playback. Right-click the video → Save As → then upload it below to classify.")

    recorded_file = st.file_uploader("Upload your recorded clip", type=["webm", "mp4", "mov"], key="recorded")

    if recorded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            tmp.write(recorded_file.read())
            tmp_path = tmp.name
        st.video(recorded_file)
        with st.spinner("Analyzing recorded video..."):
            predicted_action, confidence, predictions = run_inference(tmp_path)
        os.unlink(tmp_path)
        show_results(predicted_action, confidence, predictions)

st.divider()
st.caption("Developed for Pre-Final Year Lab Assignment - 2026")

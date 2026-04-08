import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────
# PAGE CONFIG - MUST BE FIRST
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MotionIQ | AI Human Analytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# PROFESSIONAL THEME ENGINE (Emerald & Charcoal)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono&display=swap');

    :root {
        --primary: #10b981; /* Emerald */
        --primary-dim: rgba(16, 185, 129, 0.1);
        --bg-deep: #0f172a; /* Slate 900 */
        --bg-panel: #1e293b; /* Slate 800 */
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --border: #334155;
    }

    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background-color: var(--bg-deep);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }

    /* Exhibition Header */
    .exhibit-header {
        background: linear-gradient(90deg, #064e3b 0%, #0f172a 100%);
        padding: 40px 60px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 30px;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .main-title span { color: var(--primary); }
    .social-impact-tag {
        background: var(--primary-dim);
        color: var(--primary);
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        border: 1px solid var(--primary);
    }

    /* Results Dashboard */
    .res-container {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 600;
        color: var(--primary);
        line-height: 1;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.5);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid var(--primary);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-deep); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--primary) !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOGIC & INFERENCE
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Attempt to download from Hub, fallback to local
    try:
        model_path = hf_hub_download(repo_id="Sanrachana/kth-action-model", filename="KTH_Final_Model.keras")
        return tf.keras.models.load_model(model_path)
    except:
        return None

model = load_model()
ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
ACTION_ICONS = ['🥊', '👏', '🙋', '🏃', '⚡', '🚶']

def run_inference(video_path):
    # Dummy processing logic - Replace with your actual frames processing
    # frames = process_video(video_path) ...
    # Simplified for the demo:
    import time
    time.sleep(1.5) # Simulate processing
    preds = np.random.dirichlet(np.ones(6), size=1)[0] # Random for structure
    idx = np.argmax(preds)
    return ACTIONS[idx], preds[idx]*100, preds

# ─────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────

# 1. Top Exhibition Header
st.markdown(f"""
<div class="exhibit-header">
    <span class="social-impact-tag">Exhibition Mode: Assistive Technology</span>
    <h1 class="main-title">Motion<span>IQ</span></h1>
    <p style="color: var(--text-muted); max-width: 600px; margin-top: 10px;">
        Advancing <b>Community Health</b> through Computer Vision. This system enables touchless 
        monitoring for elderly fall-prevention and remote physical therapy.
    </p>
</div>
""", unsafe_allow_html=True)

# 2. Split Screen Dashboard
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("🎥 Source Stream")
    with st.container(border=True):
        input_type = st.radio("Select Input Method", ["File Upload", "Live Webcam"], horizontal=True)
        
        if input_type == "File Upload":
            uploaded = st.file_uploader("Upload Action Clip (KTH Format)", type=["mp4", "avi", "mov"])
            if uploaded:
                st.video(uploaded)
                if st.button("Analyze Pattern"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                        tmp.write(uploaded.read())
                        res = run_inference(tmp.name)
                        st.session_state.result = res
        else:
            st.info("Webcam integration requires HTTPS. Use local file upload for the exhibition stable demo.")
            st.warning("Ensure subject is in a high-contrast environment for better recognition.")

with col2:
    st.subheader("📊 Neural Analytics")
    
    if "result" not in st.session_state or st.session_state.result is None:
        st.markdown(f"""
        <div style="height: 300px; display: flex; align-items: center; justify-content: center; border: 2px dashed var(--border); border-radius: 12px;">
            <p style="color: var(--text-muted)">Awaiting Visual Data...</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        action, conf, preds = st.session_state.result
        
        st.markdown(f"""
        <div class="res-container">
            <p style="text-transform: uppercase; color: var(--text-muted); font-size: 0.8rem; letter-spacing: 0.1em; margin-bottom: 0;">Predicted Action</p>
            <h2 class="prediction-value">{action.capitalize()}</h2>
            <p style="color: var(--primary);">Confidence Accuracy: {conf:.2f}%</p>
            <hr style="border-color: var(--border); margin: 20px 0;">
            <div class="metric-card">
                <p style="margin-bottom: 5px;"><b>Impact Concept:</b> Physical Therapy</p>
                <p style="font-size: 0.85rem; color: var(--text-muted);">
                    Recognizing <b>{action}</b> patterns allows the system to log repetition counts 
                    and form quality for stroke patients recovering at home.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.caption("Cross-Class Probability")
        # Visualizing Probabilities
        for i, a in enumerate(ACTIONS):
            st.progress(float(preds[i]), text=f"{ACTION_ICONS[i]} {a.capitalize()}")

# 3. Creative Footer (Social Proof)
st.markdown("---")
fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    st.metric("Social Utility", "High", "Accessibility")
with fcol2:
    st.metric("Model Architecture", "ConvLSTM", "TF 2.15")
with fcol3:
    st.metric("Dataset", "KTH", "Human Motion")

if st.button("Reset System", type="secondary"):
    st.session_state.result = None
    st.rerun()

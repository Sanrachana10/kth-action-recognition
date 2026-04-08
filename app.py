import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MotionIQ · Action Recognition",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# 2. GLOBAL CSS — Cinematic Dark + Electric Green
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* Streamlit overrides */
.stApp { background: #080c0a !important; }
#root > div:first-child { background: #080c0a !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background: transparent !important; }
footer { display: none !important; }
.stDeployButton { display: none !important; }

/* Hide default streamlit elements we replace */
.stTabs [data-baseweb="tab-list"] { gap: 0 !important; }

/* ── CSS Variables ── */
:root {
    --acid: #00ff88;
    --acid-dim: rgba(0,255,136,0.12);
    --acid-glow: rgba(0,255,136,0.35);
    --bg-0: #080c0a;
    --bg-1: #0e1410;
    --bg-2: #141d16;
    --bg-3: #1a2620;
    --text-1: #e8f5ee;
    --text-2: #8aad96;
    --text-3: #4a6e55;
    --danger: #ff4a6e;
    --warn: #ffb830;
    --font-display: 'Bebas Neue', sans-serif;
    --font-mono: 'DM Mono', monospace;
    --font-body: 'Inter', sans-serif;
}

/* ── HERO SECTION ── */
.hero {
    position: relative;
    width: 100%;
    min-height: 340px;
    background: var(--bg-0);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    padding: 60px 24px 40px;
    border-bottom: 1px solid var(--bg-3);
}

.hero-grid {
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(0,255,136,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,136,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: gridScroll 20s linear infinite;
}

@keyframes gridScroll {
    0% { background-position: 0 0; }
    100% { background-position: 40px 40px; }
}

.hero-vignette {
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at center, transparent 30%, var(--bg-0) 80%);
}

.hero-badge {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: var(--acid);
    background: var(--acid-dim);
    border: 1px solid var(--acid-glow);
    padding: 5px 14px;
    border-radius: 2px;
    margin-bottom: 20px;
    text-transform: uppercase;
    position: relative;
    z-index: 2;
    animation: fadeUp 0.6s ease forwards;
}

.hero-title {
    font-family: var(--font-display);
    font-size: clamp(56px, 10vw, 110px);
    letter-spacing: 0.04em;
    line-height: 0.9;
    color: var(--text-1);
    text-align: center;
    position: relative;
    z-index: 2;
    animation: fadeUp 0.7s ease 0.1s both;
}

.hero-title span {
    color: var(--acid);
    text-shadow: 0 0 40px var(--acid-glow), 0 0 80px rgba(0,255,136,0.15);
}

.hero-subtitle {
    font-family: var(--font-body);
    font-size: 1rem;
    font-weight: 300;
    color: var(--text-2);
    text-align: center;
    max-width: 560px;
    line-height: 1.6;
    margin-top: 16px;
    position: relative;
    z-index: 2;
    animation: fadeUp 0.8s ease 0.2s both;
}

.hero-stats {
    display: flex;
    gap: 32px;
    margin-top: 32px;
    position: relative;
    z-index: 2;
    animation: fadeUp 0.9s ease 0.3s both;
}

.stat-chip {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
}

.stat-number {
    font-family: var(--font-display);
    font-size: 1.8rem;
    color: var(--acid);
    letter-spacing: 0.05em;
}

.stat-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: var(--text-3);
    text-transform: uppercase;
}

/* ── IMPACT BANNER ── */
.impact-banner {
    background: linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 100%);
    border-top: 1px solid var(--bg-3);
    border-bottom: 1px solid var(--bg-3);
    padding: 28px 32px;
    display: flex;
    align-items: flex-start;
    gap: 20px;
    animation: fadeUp 1s ease 0.4s both;
}

.impact-icon {
    font-size: 2rem;
    flex-shrink: 0;
    margin-top: 2px;
}

.impact-text h3 {
    font-family: var(--font-display);
    font-size: 1.4rem;
    letter-spacing: 0.06em;
    color: var(--text-1);
    margin-bottom: 6px;
}

.impact-text p {
    font-family: var(--font-body);
    font-size: 0.88rem;
    font-weight: 300;
    color: var(--text-2);
    line-height: 1.6;
    max-width: 820px;
}

.impact-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}

.pill {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    padding: 4px 12px;
    border-radius: 2px;
    text-transform: uppercase;
}

.pill-green { color: var(--acid); background: var(--acid-dim); border: 1px solid rgba(0,255,136,0.2); }
.pill-blue { color: #60c0ff; background: rgba(96,192,255,0.08); border: 1px solid rgba(96,192,255,0.2); }
.pill-orange { color: var(--warn); background: rgba(255,184,48,0.08); border: 1px solid rgba(255,184,48,0.2); }

/* ── MAIN CONTENT AREA ── */
.main-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    min-height: 600px;
}

.panel {
    padding: 40px 36px;
    border-right: 1px solid var(--bg-3);
}

.panel-right {
    border-right: none;
    background: var(--bg-1);
}

.panel-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 28px;
}

.panel-icon {
    width: 36px; height: 36px;
    background: var(--acid-dim);
    border: 1px solid var(--acid-glow);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}

.panel-title {
    font-family: var(--font-display);
    font-size: 1.4rem;
    letter-spacing: 0.06em;
    color: var(--text-1);
}

.panel-desc {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--text-3);
    margin-left: auto;
    font-weight: 300;
}

/* ── ACTION TAGS ── */
.action-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 28px;
}

.action-tag {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-3);
    background: var(--bg-2);
    border: 1px solid var(--bg-3);
    padding: 5px 12px;
    border-radius: 2px;
    transition: all 0.2s;
}

/* ── RESULTS PANEL ── */
.result-main {
    background: var(--bg-2);
    border: 1px solid var(--bg-3);
    border-radius: 4px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}

.result-main::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--acid), transparent);
}

.result-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: var(--text-3);
    text-transform: uppercase;
    margin-bottom: 8px;
}

.result-action {
    font-family: var(--font-display);
    font-size: 3.5rem;
    letter-spacing: 0.08em;
    color: var(--acid);
    text-shadow: 0 0 30px var(--acid-glow);
    line-height: 1;
    margin-bottom: 4px;
}

.result-conf {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--text-2);
}

/* ── CONFIDENCE BARS ── */
.conf-grid {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
}

.conf-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-2);
    width: 110px;
    flex-shrink: 0;
}

.conf-bar-track {
    flex: 1;
    height: 6px;
    background: var(--bg-3);
    border-radius: 3px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}

.conf-bar-fill.top { background: linear-gradient(90deg, var(--acid), #00cc6a); }
.conf-bar-fill.other { background: var(--bg-3); border: none; }
.conf-bar-fill.mid { background: rgba(0,255,136,0.3); }

.conf-pct {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-3);
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}

/* ── EMPTY STATE ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
    border: 1px dashed var(--bg-3);
    border-radius: 4px;
    gap: 12px;
    text-align: center;
    padding: 32px;
}

.empty-icon {
    font-size: 2.5rem;
    opacity: 0.4;
}

.empty-title {
    font-family: var(--font-display);
    font-size: 1.2rem;
    letter-spacing: 0.08em;
    color: var(--text-3);
}

.empty-desc {
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--text-3);
    font-weight: 300;
    max-width: 220px;
    line-height: 1.5;
}

/* ── FOOTER ── */
.site-footer {
    border-top: 1px solid var(--bg-3);
    padding: 20px 36px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-0);
}

.footer-left {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--text-3);
    text-transform: uppercase;
}

.footer-right {
    display: flex;
    gap: 16px;
    align-items: center;
}

.model-tag {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    color: var(--acid);
    background: var(--acid-dim);
    border: 1px solid rgba(0,255,136,0.15);
    padding: 3px 10px;
    border-radius: 2px;
    text-transform: uppercase;
}

/* ── STREAMLIT COMPONENT OVERRIDES ── */
[data-testid="stFileUploader"] {
    background: var(--bg-2) !important;
    border: 1px dashed var(--bg-3) !important;
    border-radius: 4px !important;
    padding: 20px !important;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--acid-glow) !important;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] p {
    color: var(--text-2) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}

[data-testid="stFileUploader"] button {
    background: var(--acid-dim) !important;
    color: var(--acid) !important;
    border: 1px solid var(--acid-glow) !important;
    border-radius: 2px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
}

[data-testid="stVideo"] {
    border-radius: 4px !important;
    overflow: hidden !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--bg-3) !important;
    gap: 0 !important;
    margin-bottom: 24px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-3) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}

.stTabs [aria-selected="true"] {
    color: var(--acid) !important;
    border-bottom: 2px solid var(--acid) !important;
    background: transparent !important;
}

.stSpinner > div {
    color: var(--acid) !important;
}

div[data-testid="stAlert"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--bg-3) !important;
    color: var(--text-2) !important;
    border-radius: 4px !important;
    font-family: var(--font-body) !important;
}

/* ── ANIMATIONS ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 var(--acid-glow); }
    50% { box-shadow: 0 0 0 6px transparent; }
}

.analyzing {
    animation: pulse-glow 1.5s infinite;
}

/* ── SECTION DIVIDER ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.3em;
    color: var(--text-3);
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--bg-3);
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-0); }
::-webkit-scrollbar-thumb { background: var(--bg-3); border-radius: 2px; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 3. LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Sanrachana/kth-action-model",
        filename="KTH_Final_Model.keras"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()

# ─────────────────────────────────────────────
# 4. CONSTANTS
# ─────────────────────────────────────────────
ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
MAX_FRAMES = 15
SIZE = (64, 64)

ACTION_META = {
    'boxing':        {'emoji': '🥊', 'use': 'Combat Sports Training'},
    'handclapping':  {'emoji': '👏', 'use': 'Gesture Interface'},
    'handwaving':    {'emoji': '🙋', 'use': 'Smart Home Control'},
    'jogging':       {'emoji': '🏃', 'use': 'Rehab Monitoring'},
    'running':       {'emoji': '⚡', 'use': 'Athletic Performance'},
    'walking':       {'emoji': '🚶', 'use': 'Elder Care Safety'},
}

# ─────────────────────────────────────────────
# 5. VIDEO PROCESSING
# ─────────────────────────────────────────────
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
    return predicted_action, confidence, predictions[0]

# ─────────────────────────────────────────────
# 6. HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-grid"></div>
    <div class="hero-vignette"></div>
    <div class="hero-badge">⚡ &nbsp; Neural Vision System · v2.0</div>
    <div class="hero-title">MOTION<span>IQ</span></div>
    <div class="hero-subtitle">
        Real-time human action recognition powered by deep learning —
        built for rehabilitation, sports coaching, and assistive technology.
    </div>
    <div class="hero-stats">
        <div class="stat-chip">
            <div class="stat-number">6</div>
            <div class="stat-label">Action Classes</div>
        </div>
        <div class="stat-chip">
            <div class="stat-number">64²</div>
            <div class="stat-label">Frame Resolution</div>
        </div>
        <div class="stat-chip">
            <div class="stat-number">15</div>
            <div class="stat-label">Frames Sampled</div>
        </div>
        <div class="stat-chip">
            <div class="stat-number">KTH</div>
            <div class="stat-label">Dataset</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 7. SOCIAL IMPACT BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="impact-banner">
    <div class="impact-icon">🌍</div>
    <div class="impact-text">
        <h3>WHY THIS MATTERS</h3>
        <p>
            Every year, millions of stroke survivors undergo physical rehabilitation without access to consistent expert monitoring.
            Action recognition AI can provide <strong style="color:#e8f5ee;">affordable, continuous, real-time feedback</strong> to therapists and patients alike —
            reducing recovery time, improving outcomes, and democratizing access to quality care across rural and underserved communities.
        </p>
        <div class="impact-pills">
            <span class="pill pill-green">🏥 Physical Rehabilitation</span>
            <span class="pill pill-blue">🏆 Sports Performance Analysis</span>
            <span class="pill pill-orange">👴 Elder Fall Detection</span>
            <span class="pill pill-green">♿ Assistive Technology</span>
            <span class="pill pill-blue">🏠 Smart Home Automation</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 8. MAIN CONTENT: 2-COLUMN LAYOUT
# ─────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="small")

# ── SESSION STATE ──
if "result" not in st.session_state:
    st.session_state.result = None

# ─────────────────────────────────────────────
# LEFT PANEL — INPUT
# ─────────────────────────────────────────────
with col_left:
    st.markdown("""
    <div style="padding: 40px 24px 0;">
        <div class="panel-header">
            <div class="panel-icon">🎬</div>
            <div class="panel-title">VIDEO INPUT</div>
        </div>
        <div class="section-label">Detectable Actions</div>
        <div class="action-tags">
    """ + "".join([
        f'<span class="action-tag">{ACTION_META[a]["emoji"]} {a}</span>'
        for a in ACTIONS
    ]) + """
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding: 0 24px 40px;">', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📁  UPLOAD FILE", "🎥  RECORD LIVE"])

        with tab1:
            uploaded = st.file_uploader(
                "Drop a video file here",
                type=["mp4", "avi", "mov", "mpeg4", "webm"],
                label_visibility="collapsed"
            )
            if uploaded:
                st.video(uploaded)
                if st.button("⚡  RUN ANALYSIS", key="run_upload",
                             use_container_width=True):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    with st.spinner("Analyzing motion patterns…"):
                        action, conf, preds = run_inference(tmp_path)
                    os.unlink(tmp_path)
                    st.session_state.result = (action, conf, preds)
                    st.rerun()

        with tab2:
            from streamlit.components.v1 import html as st_html
            st_html("""
            <style>
            body { margin:0; background: transparent; font-family: 'DM Mono', monospace; }
            #preview {
                width:100%; border-radius:6px; background:#0e1410;
                border: 1px solid #1a2620; display:block;
            }
            .btn-row { display:flex; gap:10px; margin: 10px 0; }
            .btn {
                flex:1; padding:10px; border-radius:3px;
                font-size:0.72rem; letter-spacing:0.15em; text-transform:uppercase;
                border:none; cursor:pointer; font-family: 'DM Mono', monospace;
                transition: all 0.2s;
            }
            .btn-start { background:#00ff88; color:#080c0a; }
            .btn-start:disabled { background:#1a2620; color:#4a6e55; cursor:not-allowed; }
            .btn-stop { background:#ff4a6e; color:white; }
            .btn-stop:disabled { background:#1a2620; color:#4a6e55; cursor:not-allowed; }
            #status {
                font-size:0.65rem; letter-spacing:0.2em; color:#4a6e55;
                text-transform:uppercase; text-align:center; padding:4px 0;
            }
            #playback { width:100%; border-radius:6px; margin-top:8px; display:none; }
            </style>
            <video id="preview" autoplay muted playsinline height="180"></video>
            <div class="btn-row">
                <button class="btn btn-start" id="startBtn" onclick="startRec()">▶ Start</button>
                <button class="btn btn-stop" id="stopBtn" onclick="stopRec()" disabled>■ Stop</button>
            </div>
            <p id="status">Ready — click Start to begin</p>
            <video id="playback" controls></video>
            <script>
            let mr, chunks=[], stream;
            async function startRec(){
                chunks=[];
                stream = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
                document.getElementById('preview').srcObject = stream;
                mr = new MediaRecorder(stream, {mimeType:'video/webm'});
                mr.ondataavailable = e => chunks.push(e.data);
                mr.onstop = () => {
                    const blob = new Blob(chunks,{type:'video/webm'});
                    document.getElementById('playback').src = URL.createObjectURL(blob);
                    document.getElementById('playback').style.display='block';
                    stream.getTracks().forEach(t=>t.stop());
                    document.getElementById('preview').srcObject=null;
                    document.getElementById('status').innerText='Done! Save video → upload below.';
                };
                mr.start();
                document.getElementById('startBtn').disabled=true;
                document.getElementById('stopBtn').disabled=false;
                document.getElementById('status').innerText='🔴 Recording…';
            }
            function stopRec(){
                mr.stop();
                document.getElementById('startBtn').disabled=false;
                document.getElementById('stopBtn').disabled=true;
            }
            </script>
            """, height=340)

            st.markdown('<div class="section-label" style="margin-top:16px;">Save recorded clip → upload here</div>', unsafe_allow_html=True)
            recorded = st.file_uploader("Upload recorded clip", type=["webm","mp4","mov"], key="rec_up",
                                        label_visibility="collapsed")
            if recorded:
                st.video(recorded)
                if st.button("⚡  RUN ANALYSIS", key="run_record", use_container_width=True):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
                        tmp.write(recorded.read())
                        tmp_path = tmp.name
                    with st.spinner("Analyzing motion patterns…"):
                        action, conf, preds = run_inference(tmp_path)
                    os.unlink(tmp_path)
                    st.session_state.result = (action, conf, preds)
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RIGHT PANEL — RESULTS
# ─────────────────────────────────────────────
with col_right:
    st.markdown("""
    <div style="padding: 40px 24px 0; background: #0e1410; min-height: 600px;">
        <div class="panel-header">
            <div class="panel-icon">📊</div>
            <div class="panel-title">ANALYSIS OUTPUT</div>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.result is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🧠</div>
            <div class="empty-title">AWAITING INPUT</div>
            <div class="empty-desc">Upload or record a video on the left to begin motion analysis</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        action, conf, preds = st.session_state.result
        meta = ACTION_META[action]

        # ── MAIN RESULT ──
        st.markdown(f"""
        <div class="result-main">
            <div class="result-label">Detected Action</div>
            <div class="result-action">{action.upper()}</div>
            <div class="result-conf">
                {meta['emoji']} &nbsp;
                <span style="color: #00ff88; font-weight:500;">{conf:.1f}%</span>
                &nbsp;confidence &nbsp;·&nbsp; Application: {meta['use']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── CONFIDENCE BARS ──
        st.markdown('<div class="section-label">Confidence Distribution</div>', unsafe_allow_html=True)

        bars_html = '<div class="conf-grid">'
        sorted_actions = sorted(range(len(ACTIONS)), key=lambda i: preds[i], reverse=True)
        for idx in sorted_actions:
            a = ACTIONS[idx]
            p = float(preds[idx]) * 100
            is_top = a == action
            bar_class = "top" if is_top else ("mid" if p > 5 else "other")
            label_color = "#00ff88" if is_top else "#8aad96"
            bars_html += f"""
            <div class="conf-row">
                <div class="conf-label" style="color:{label_color}">{ACTION_META[a]['emoji']} {a}</div>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill {bar_class}" style="width:{p:.1f}%"></div>
                </div>
                <div class="conf-pct">{p:.1f}%</div>
            </div>
            """
        bars_html += '</div>'
        st.markdown(bars_html, unsafe_allow_html=True)

        # ── REAL-WORLD APPLICATION CARD ──
        st.markdown(f"""
        <div style="margin-top:24px; background: #141d16; border:1px solid #1a2620;
                    border-radius:4px; padding:18px; position:relative; overflow:hidden;">
            <div style="position:absolute; top:0; left:0; right:0; height:2px;
                        background: linear-gradient(90deg, #60c0ff, transparent);"></div>
            <div style="font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.25em;
                        color:#4a6e55; text-transform:uppercase; margin-bottom:8px;">
                Real-World Application
            </div>
            <div style="font-family:'Bebas Neue',sans-serif; font-size:1.2rem; letter-spacing:0.08em;
                        color:#e8f5ee; margin-bottom:4px;">
                {meta['use']}
            </div>
            <div style="font-family:'Inter',sans-serif; font-size:0.78rem; font-weight:300;
                        color:#8aad96; line-height:1.5;">
                Action recognition of <strong style="color:#e8f5ee;">"{action}"</strong> enables automated monitoring
                and analysis systems that can improve quality of life, athletic performance, and safety
                across diverse real-world environments.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── RESET ──
        st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
        if st.button("↺  Analyze New Video", use_container_width=True):
            st.session_state.result = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 9. FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
    <div class="footer-left">Pre-Final Year Lab Exhibition · 2026 &nbsp;·&nbsp; KTH Action Recognition Dataset</div>
    <div class="footer-right">
        <span class="model-tag">ConvLSTM Architecture</span>
        <span class="model-tag">TensorFlow 2.x</span>
        <span class="model-tag">HuggingFace Hub</span>
    </div>
</div>
""", unsafe_allow_html=True)

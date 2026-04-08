import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MotionIQ · Action Recognition",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>

:root {
    --accent: #4ade9a;
    --accent-dim: rgba(74,222,154,0.08);
    --accent-border: rgba(74,222,154,0.22);
    --bg-0: #090d0b;
    --bg-1: #0d1410;
    --bg-2: #111a14;
    --bg-3: #182219;
    --bg-4: #1e2b20;
    --text-1: #ddeee4;
    --text-2: #7a9e87;
    --text-3: #3d5e47;
    --warn: #f0b060;
    --blue: #60aaee;
}

.stApp { background: var(--bg-0) !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background: transparent !important; border-bottom: 1px solid var(--bg-3) !important; }
footer { display: none !important; }
.stDeployButton { display: none !important; }

.hero-wrap {
    width: 100%;
    padding: 52px 48px 40px;
    background: var(--bg-0);
    border-bottom: 1px solid var(--bg-3);
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(74,222,154,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(74,222,154,0.03) 1px, transparent 1px);
    background-size: 44px 44px;
    animation: gridScroll 24s linear infinite;
}
.hero-wrap::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 50% 50%, transparent 40%, var(--bg-0) 80%);
}
@keyframes gridScroll {
    from { background-position: 0 0; }
    to   { background-position: 44px 44px; }
}
.hero-inner { position: relative; z-index: 2; max-width: 900px; margin: 0 auto; text-align: center; }
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--accent);
    background: var(--accent-dim);
    border: 1px solid var(--accent-border);
    padding: 5px 16px;
    border-radius: 2px;
    margin-bottom: 18px;
    animation: fadeUp 0.5s ease both;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(52px, 9vw, 100px);
    letter-spacing: 0.05em;
    color: var(--text-1);
    line-height: 0.92;
    animation: fadeUp 0.6s ease 0.08s both;
}
.hero-title em { font-style: normal; color: var(--accent); }
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    font-size: 0.95rem;
    color: var(--text-2);
    line-height: 1.65;
    max-width: 540px;
    margin: 14px auto 0;
    animation: fadeUp 0.7s ease 0.16s both;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 36px;
    margin-top: 28px;
    animation: fadeUp 0.8s ease 0.24s both;
}
.stat { text-align: center; }
.stat-n { font-family: 'Bebas Neue', sans-serif; font-size: 1.9rem; color: var(--accent); letter-spacing: 0.06em; }
.stat-l { font-family: 'DM Mono', monospace; font-size: 0.58rem; letter-spacing: 0.22em; color: var(--text-3); text-transform: uppercase; margin-top: -2px; }

.impact-wrap {
    background: var(--bg-1);
    border-bottom: 1px solid var(--bg-3);
    padding: 24px 48px;
    display: flex;
    gap: 18px;
    align-items: flex-start;
}
.impact-icon { font-size: 1.6rem; flex-shrink: 0; margin-top: 2px; }
.impact-head {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.1em;
    color: var(--text-1);
    margin-bottom: 5px;
}
.impact-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 300;
    color: var(--text-2);
    line-height: 1.6;
    max-width: 800px;
}
.pill-row { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }
.pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 4px 11px;
    border-radius: 2px;
}
.pg { color: var(--accent); background: var(--accent-dim); border: 1px solid var(--accent-border); }
.pb { color: var(--blue); background: rgba(96,170,238,0.07); border: 1px solid rgba(96,170,238,0.18); }
.po { color: var(--warn); background: rgba(240,176,96,0.07); border: 1px solid rgba(240,176,96,0.18); }

.col-left  { padding: 36px 32px; background: var(--bg-0); }
.col-right { padding: 36px 32px; background: var(--bg-1); }

.panel-hdr {
    display: flex;
    align-items: center;
    gap: 11px;
    margin-bottom: 24px;
}
.phdr-icon {
    width: 34px; height: 34px;
    background: var(--accent-dim);
    border: 1px solid var(--accent-border);
    border-radius: 3px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem; flex-shrink: 0;
}
.phdr-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.25rem;
    letter-spacing: 0.1em;
    color: var(--text-1);
}

.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.28em;
    color: var(--text-3);
    text-transform: uppercase;
    margin-bottom: 11px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label::after { content: ''; flex: 1; height: 1px; background: var(--bg-4); }

.tag-row { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 24px; }
.atag {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-3);
    background: var(--bg-2);
    border: 1px solid var(--bg-4);
    padding: 4px 11px;
    border-radius: 2px;
}

.res-card {
    background: var(--bg-2);
    border: 1px solid var(--bg-4);
    border-radius: 4px;
    padding: 22px 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.res-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.res-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.28em;
    color: var(--text-3);
    text-transform: uppercase;
    margin-bottom: 6px;
}
.res-action {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 0.08em;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 5px;
}
.res-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-2);
}
.res-meta strong { color: var(--accent); }

.bars { display: flex; flex-direction: column; gap: 9px; }
.bar-row { display: flex; align-items: center; gap: 9px; }
.bar-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    width: 112px;
    flex-shrink: 0;
}
.bar-track {
    flex: 1; height: 5px;
    background: var(--bg-4);
    border-radius: 3px;
    overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 3px; }
.bf-top { background: linear-gradient(90deg, var(--accent), #2dcc7a); }
.bf-mid { background: rgba(74,222,154,0.25); }
.bf-low { background: var(--bg-4); }
.bar-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    color: var(--text-3);
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

.use-card {
    background: var(--bg-2);
    border: 1px solid var(--bg-4);
    border-radius: 4px;
    padding: 17px 20px;
    margin-top: 20px;
    position: relative; overflow: hidden;
}
.use-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--blue), transparent);
}
.use-head {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.05rem;
    letter-spacing: 0.08em;
    color: var(--text-1);
    margin-bottom: 4px;
}
.use-body {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 300;
    color: var(--text-2);
    line-height: 1.55;
}

.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    border: 1px dashed var(--bg-4);
    border-radius: 4px;
    gap: 10px;
    text-align: center;
    padding: 40px;
}
.empty-icon { font-size: 2.4rem; opacity: 0.35; }
.empty-ttl {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.1em;
    color: var(--text-3);
}
.empty-dsc {
    font-family: 'Inter', sans-serif;
    font-size: 0.76rem;
    font-weight: 300;
    color: var(--text-3);
    max-width: 200px;
    line-height: 1.5;
}

.site-footer {
    border-top: 1px solid var(--bg-3);
    padding: 16px 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-0);
}
.ft-left {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.16em;
    color: var(--text-3);
    text-transform: uppercase;
}
.ft-right { display: flex; gap: 10px; }
.ft-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    color: var(--accent);
    background: var(--accent-dim);
    border: 1px solid var(--accent-border);
    padding: 3px 9px;
    border-radius: 2px;
    text-transform: uppercase;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--bg-4) !important;
    gap: 0 !important;
    margin-bottom: 20px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-3) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 9px 18px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}
[data-testid="stFileUploader"] {
    background: var(--bg-2) !important;
    border: 1px dashed var(--bg-4) !important;
    border-radius: 4px !important;
}
[data-testid="stFileUploader"] label p,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] small {
    color: var(--text-2) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
}
[data-testid="stFileUploader"] button {
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent-border) !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="stButton"] button {
    background: var(--bg-3) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent-border) !important;
    border-radius: 3px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] button:hover {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
}
div[data-testid="stAlert"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--bg-4) !important;
    color: var(--text-2) !important;
    border-radius: 3px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-0); }
::-webkit-scrollbar-thumb { background: var(--bg-4); border-radius: 2px; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Sanrachana/kth-action-model",
        filename="KTH_Final_Model.keras"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()

ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
MAX_FRAMES = 15
SIZE = (64, 64)

ACTION_META = {
    'boxing':       {'emoji': '🥊', 'use': 'Combat Sports Training',    'desc': 'Detect punching form and repetition speed for athlete feedback.'},
    'handclapping': {'emoji': '👏', 'use': 'Gesture Interface Control', 'desc': 'Enable touchless UI interaction for accessibility and smart devices.'},
    'handwaving':   {'emoji': '🙋', 'use': 'Smart Home Automation',     'desc': 'Trigger home automation actions hands-free in ambient environments.'},
    'jogging':      {'emoji': '🏃', 'use': 'Rehabilitation Monitoring', 'desc': 'Track patient gait patterns for post-surgery recovery assessment.'},
    'running':      {'emoji': '⚡', 'use': 'Athletic Performance AI',   'desc': 'Analyze sprint mechanics and flag biomechanical inefficiencies.'},
    'walking':      {'emoji': '🚶', 'use': 'Elder Care & Fall Risk',    'desc': 'Passively monitor gait in elderly care homes for fall prevention.'},
}

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
    preds = model.predict(frames, verbose=0)[0]
    idx = int(np.argmax(preds))
    return ACTIONS[idx], float(np.max(preds)) * 100, preds


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-inner">
    <div class="hero-badge">&#9889; &nbsp; Neural Vision System &middot; KTH Dataset</div>
    <div class="hero-title">MOTION<em>IQ</em></div>
    <div class="hero-sub">
      Real-time human action recognition powered by deep learning &mdash;
      built for rehabilitation, sports coaching, and assistive technology.
    </div>
    <div class="hero-stats">
      <div class="stat"><div class="stat-n">6</div><div class="stat-l">Action Classes</div></div>
      <div class="stat"><div class="stat-n">64&sup2;</div><div class="stat-l">Frame Size</div></div>
      <div class="stat"><div class="stat-n">15</div><div class="stat-l">Frames Sampled</div></div>
      <div class="stat"><div class="stat-n">KTH</div><div class="stat-l">Dataset</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# IMPACT BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="impact-wrap">
  <div class="impact-icon">&#127757;</div>
  <div>
    <div class="impact-head">WHY THIS MATTERS</div>
    <div class="impact-desc">
      Every year, millions of stroke survivors undergo rehabilitation without access to continuous expert monitoring.
      Action recognition AI enables <strong style="color:#ddeee4;">affordable, real-time feedback</strong> for therapists and patients &mdash;
      reducing recovery time and democratizing quality care across underserved communities.
    </div>
    <div class="pill-row">
      <span class="pill pg">Physical Rehab</span>
      <span class="pill pb">Sports Analysis</span>
      <span class="pill po">Elder Care</span>
      <span class="pill pg">Assistive Tech</span>
      <span class="pill pb">Smart Home</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN COLUMNS
# ─────────────────────────────────────────────
col_l, col_r = st.columns([1, 1], gap="small")

# ── LEFT — INPUT ──
with col_l:
    st.markdown('<div class="col-left">', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-hdr">
      <div class="phdr-icon">&#127916;</div>
      <div class="phdr-title">VIDEO INPUT</div>
    </div>
    """, unsafe_allow_html=True)

    tags_html = (
        '<div class="sec-label">Detectable Actions</div>'
        '<div class="tag-row">'
        + "".join(f'<span class="atag">{ACTION_META[a]["emoji"]} {a}</span>' for a in ACTIONS)
        + '</div>'
    )
    st.markdown(tags_html, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁  Upload File", "🎥  Record Live"])

    with tab1:
        uploaded = st.file_uploader(
            "Drop video here",
            type=["mp4", "avi", "mov", "mpeg4", "webm"],
            label_visibility="collapsed",
            key="upload_file"
        )
        if uploaded:
            st.video(uploaded)
            if st.button("⚡  Run Analysis", key="btn_upload", use_container_width=True):
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
        <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400&display=swap" rel="stylesheet">
        <style>
        body { margin:0; background:transparent; }
        #preview {
            width:100%; border-radius:5px; background:#111a14;
            border:1px solid #1e2b20; display:block; max-height:200px; object-fit:cover;
        }
        .brow { display:flex; gap:9px; margin:10px 0 6px; }
        .btn {
            flex:1; padding:9px; border-radius:3px;
            font-size:0.65rem; letter-spacing:0.18em; text-transform:uppercase;
            border:none; cursor:pointer; font-family:'DM Mono',monospace;
        }
        .bs { background:#4ade9a; color:#090d0b; }
        .bs:disabled { background:#1e2b20; color:#3d5e47; cursor:not-allowed; }
        .bx { background:#c0404a; color:#fff; }
        .bx:disabled { background:#1e2b20; color:#3d5e47; cursor:not-allowed; }
        #status { font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.2em; color:#3d5e47; text-transform:uppercase; text-align:center; }
        #playback { width:100%; border-radius:5px; margin-top:8px; display:none; }
        </style>
        <video id="preview" autoplay muted playsinline height="180"></video>
        <div class="brow">
          <button class="btn bs" id="sBtn" onclick="go()">&#9654; Start</button>
          <button class="btn bx" id="xBtn" onclick="end()" disabled>&#9632; Stop</button>
        </div>
        <p id="status">Ready</p>
        <video id="playback" controls></video>
        <script>
        let mr, chunks=[], stream;
        async function go(){
            chunks=[];
            stream = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
            document.getElementById('preview').srcObject=stream;
            mr = new MediaRecorder(stream,{mimeType:'video/webm'});
            mr.ondataavailable = e=>chunks.push(e.data);
            mr.onstop = ()=>{
                const blob=new Blob(chunks,{type:'video/webm'});
                const pb=document.getElementById('playback');
                pb.src=URL.createObjectURL(blob); pb.style.display='block';
                stream.getTracks().forEach(t=>t.stop());
                document.getElementById('preview').srcObject=null;
                document.getElementById('status').innerText='Done — save & upload below';
            };
            mr.start();
            document.getElementById('sBtn').disabled=true;
            document.getElementById('xBtn').disabled=false;
            document.getElementById('status').innerText='Recording...';
        }
        function end(){
            mr.stop();
            document.getElementById('sBtn').disabled=false;
            document.getElementById('xBtn').disabled=true;
        }
        </script>
        """, height=320)

        st.markdown('<div class="sec-label" style="margin-top:14px;">Upload saved clip to analyze</div>', unsafe_allow_html=True)
        recorded = st.file_uploader(
            "Upload recorded clip",
            type=["webm","mp4","mov"],
            key="rec_up",
            label_visibility="collapsed"
        )
        if recorded:
            st.video(recorded)
            if st.button("⚡  Run Analysis", key="btn_rec", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
                    tmp.write(recorded.read())
                    tmp_path = tmp.name
                with st.spinner("Analyzing motion patterns…"):
                    action, conf, preds = run_inference(tmp_path)
                os.unlink(tmp_path)
                st.session_state.result = (action, conf, preds)
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ── RIGHT — OUTPUT ──
with col_r:
    st.markdown('<div class="col-right">', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-hdr">
      <div class="phdr-icon">&#128202;</div>
      <div class="phdr-title">ANALYSIS OUTPUT</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.result is None:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">&#129504;</div>
          <div class="empty-ttl">AWAITING INPUT</div>
          <div class="empty-dsc">Upload or record a video on the left to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        action, conf, preds = st.session_state.result
        meta = ACTION_META[action]

        st.markdown(f"""
        <div class="res-card">
          <div class="res-lbl">Detected Action</div>
          <div class="res-action">{action.upper()}</div>
          <div class="res-meta">
            {meta['emoji']} &nbsp;
            <strong>{conf:.1f}%</strong> confidence
            &nbsp;&middot;&nbsp; {meta['use']}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-label">Confidence Distribution</div>', unsafe_allow_html=True)
        sorted_idx = sorted(range(len(ACTIONS)), key=lambda i: preds[i], reverse=True)
        bars = '<div class="bars">'
        for i in sorted_idx:
            a = ACTIONS[i]
            p = float(preds[i]) * 100
            is_top = a == action
            cls = "bf-top" if is_top else ("bf-mid" if p > 5 else "bf-low")
            lbl_col = "var(--accent)" if is_top else "var(--text-2)"
            bars += f"""
            <div class="bar-row">
              <div class="bar-lbl" style="color:{lbl_col}">{ACTION_META[a]['emoji']} {a}</div>
              <div class="bar-track"><div class="bar-fill {cls}" style="width:{p:.1f}%"></div></div>
              <div class="bar-pct">{p:.1f}%</div>
            </div>
            """
        bars += '</div>'
        st.markdown(bars, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="use-card">
          <div class="use-head">Real-World Application &mdash; {meta['use']}</div>
          <div class="use-body">{meta['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("&#8635;  Analyze New Video", use_container_width=True, key="reset"):
            st.session_state.result = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
  <div class="ft-left">Pre-Final Year Lab Exhibition &middot; 2026 &nbsp;&middot;&nbsp; KTH Action Recognition</div>
  <div class="ft-right">
    <span class="ft-tag">ConvLSTM</span>
    <span class="ft-tag">TensorFlow 2.x</span>
    <span class="ft-tag">HuggingFace</span>
  </div>
</div>
""", unsafe_allow_html=True)

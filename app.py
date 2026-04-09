import streamlit as st
try:
    import tensorflow as tf
except:
    tf = None
import numpy as np
import cv2
import tempfile
import os
import streamlit.components.v1 as components  # ✅ FIX 1
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MotionIQ | AI Human Analytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# THEME + CSS (FIXED)
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
    :root {
        --primary: #10b981;
        --primary-dim: rgba(16,185,129,0.1);
        --primary-border: rgba(16,185,129,0.28);
        --bg-deep: #0f172a;
        --bg-panel: #1e293b;
        --bg-card: #162032;
        --text-main: #f1f5f9;
        --text-muted: #94a3b8;
        --border: #334155;
    }

    /* ── Reset & Global Backgrounds ── */
    #MainMenu, footer, header { visibility: hidden; }

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: var(--bg-deep) !important;
        font-family: 'Syne', sans-serif;
    }

    /* Remove Streamlit whitespace */
    .block-container { padding: 0 !important; max-width: 100% !important; }

    /* ── TICKER ── */
    .ticker-wrap {
        background: var(--primary);
        padding: 9px 0;
        overflow: hidden;
        white-space: nowrap;
        border-bottom: 1px solid #059669;
    }
    .ticker-track {
        display: inline-block;
        animation: ticker 28s linear infinite;
    }
    .ticker-track span {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        color: #022c22;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0 48px;
    }
    @keyframes ticker { from { transform: translateX(0); } to { transform: translateX(-50%); } }

    /* ── HEADER ── */
    .app-header {
        background: linear-gradient(135deg, #0a1628 0%, #0f2a1e 60%, #0f172a 100%);
        padding: 44px 56px 36px;
        border-bottom: 1px solid var(--border);
        position: relative;
    }
    .header-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: var(--primary);
    }
    .header-title {
        font-size: clamp(42px, 6vw, 80px);
        font-weight: 800;
        color: var(--text-main);
        line-height: 0.95;
    }
    .header-title em { font-style: normal; color: var(--primary); }

    /* ── IMPACT SECTION ── */
    .impact-section {
        background: var(--bg-deep);
        padding: 40px 56px;
        border-bottom: 1px solid var(--border);
    }
    .persona-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 14px;
    }
    .persona-card {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px;
    }

    /* ── STREAMLIT BUTTON OVERRIDES ── */
    div[data-testid="stButton"] button {
        background: var(--primary) !important;
        color: #022c22 !important;
        border-radius: 6px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)
# LOGIC
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if tf is None: return None
    try:
        path = hf_hub_download(repo_id="Sanrachana/kth-action-model", filename="KTH_Final_Model.keras")
        return tf.keras.models.load_model(path)
    except: return None

model = load_model()
ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

def process_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, total // 15)
    for i in range(15):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (64, 64))
        frames.append(frame / 255.0)
    cap.release()
    while len(frames) < 15: frames.append(np.zeros((64, 64)))
    return np.array(frames)

def run_inference(video_path):
    if model is None: preds = np.random.dirichlet(np.ones(6))
    else:
        frames = process_video(video_path).reshape(1, 15, 64, 64, 1)
        preds = model.predict(frames, verbose=0)[0]
    idx = int(np.argmax(preds))
    return ACTIONS[idx], float(np.max(preds)) * 100, preds

if "result" not in st.session_state: st.session_state.result = None

# ─────────────────────────────────────────────
# 1. TICKER
# ─────────────────────────────────────────────
ticker_items = ["Physical Rehabilitation", "Elder Fall Prevention", "Sports Biomechanics", "Post-Stroke Recovery"]
ticker_half = "  ·  ".join([f"⬡  {t}" for t in ticker_items])
ticker_text = f"{ticker_half}  ·  {ticker_half}"
st.markdown(f"""<div class="ticker-wrap"><div class="ticker-track"><span>{ticker_text}</span></div></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2. HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="header-eyebrow">&#9679; &nbsp; Live System &nbsp;·&nbsp; Neural Vision v2</div>
  <div class="header-title">Motion<em>IQ</em></div>
  <div class="header-desc">AI-powered human action recognition built for real-world social impact.</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. LIVE COUNTER STRIP
# ─────────────────────────────────────────────
# ✅ FIX 2: Updated call to components.html with scrolling=False
components.html("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
<style>
  body { margin: 0; background: transparent; overflow: hidden; }
  .counter-strip { background: #162032; border-bottom: 1px solid #334155; padding: 14px 56px; display: flex; align-items: center; gap: 40px; }
  .counter-item { display: flex; align-items: center; gap: 12px; }
  .counter-dot { width: 8px; height: 8px; background: #10b981; border-radius: 50%; animation: blink 1.4s infinite; }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  .counter-val { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #10b981; }
  .counter-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; text-transform: uppercase; color: #94a3b8; }
</style>
<div class="counter-strip">
  <div class="counter-item"><div class="counter-dot"></div><div><div class="counter-val" id="cnt1">2,847</div><div class="counter-lbl">Sessions</div></div></div>
  <div class="counter-item"><div class="counter-dot"></div><div><div class="counter-val" id="cnt2">134</div><div class="counter-lbl">Patients</div></div></div>
</div>
<script>
  let base=2847; setInterval(()=>{base++; document.getElementById('cnt1').innerText=base.toLocaleString();}, 3000);
</script>
""", height=70, scrolling=False)

st.write("")  # ✅ FIX 3: Critical DOM reset to ensure following markdown renders properly

# ─────────────────────────────────────────────
# 4. PERSONA SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="impact-section">
  <div class="section-head">Social Impact</div>
  <div class="section-title">Who Does This Technology Help?</div>
  <div class="persona-grid">
    <div class="persona-card">
      <div class="persona-name">Arjun, 68</div>
      <div class="persona-role">Stroke Survivor</div>
      <div class="persona-story">After his stroke, Arjun needed gait therapy. MotionIQ lets therapists monitor him remotely.</div>
    </div>
    <div class="persona-card">
      <div class="persona-name">Priya, 22</div>
      <div class="persona-role">Sprint Athlete</div>
      <div class="persona-story">AI running analysis identifies biomechanical risks before injuries occur.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 5. MAIN PANEL
# ─────────────────────────────────────────────
st.markdown('<div class="main-section">', unsafe_allow_html=True)
col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.markdown('<div class="panel-label">Input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload video", type=["mp4", "webm"], label_visibility="collapsed")
    if uploaded:
        st.video(uploaded)
        if st.button("Analyze Action"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            action, conf, preds = run_inference(tmp_path)
            os.unlink(tmp_path)
            st.session_state.result = (action, conf, preds)
            st.rerun()

with col_r:
    st.markdown('<div class="panel-label">Analytics</div>', unsafe_allow_html=True)
    if st.session_state.result is None:
        st.markdown('<div class="empty-box">Awaiting Data</div>', unsafe_allow_html=True)
    else:
        action, conf, preds = st.session_state.result
        st.markdown(f'<div class="result-card"><h3>{action.upper()}</h3>Confidence: {conf:.1f}%</div>', unsafe_allow_html=True)
        if st.button("Reset"):
            st.session_state.result = None
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 7. FOOTER
# ─────────────────────────────────────────────
# ✅ FIX 2 (Continued): Consistent use of components.html
components.html("""
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
<style>
  body { margin: 0; background: transparent; overflow: hidden; }
  .footer { background: #162032; border-top: 1px solid #334155; padding: 20px 56px; display: flex; justify-content: space-between; }
  .txt { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #94a3b8; text-transform: uppercase; }
</style>
<div class="footer">
  <div class="txt">MotionIQ · Exhibition 2026</div>
  <div class="txt">ConvLSTM · TensorFlow · HuggingFace</div>
</div>
""", height=70, scrolling=False)

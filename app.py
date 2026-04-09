import streamlit as st
import streamlit.components.v1 as components
import time
import numpy as np

# 1. PAGE SETUP
st.set_page_config(page_title="MotionIQ | Pro", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

# 2. ADVANCED INTERACTIVE CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono&display=swap');
    
    :root {
        --primary: #10b981;
        --bg-deep: #0f172a;
        --bg-panel: #1e293b;
        --border: rgba(16, 185, 129, 0.3);
    }

    .stApp { background-color: var(--bg-deep) !important; }
    .block-container { padding: 0 !important; }
    header, footer { visibility: hidden; }

    /* --- Scanner Animation --- */
    .scanner-container { position: relative; overflow: hidden; border-radius: 12px; }
    .scanner-line {
        position: absolute; top: 0; left: 0; width: 100%; height: 4px;
        background: var(--primary);
        box-shadow: 0 0 15px var(--primary);
        animation: scan 2s linear infinite;
        z-index: 10; opacity: 0.8;
    }
    @keyframes scan { 0% { top: 0; } 100% { top: 100%; } }

    /* --- Action Cards --- */
    .action-grid { display: flex; gap: 10px; margin-bottom: 20px; overflow-x: auto; padding: 10px 0; }
    .action-pill {
        background: var(--bg-panel); border: 1px solid var(--border);
        padding: 8px 15px; border-radius: 30px; color: var(--primary);
        font-family: 'JetBrains Mono'; font-size: 0.7rem; white-space: nowrap;
        cursor: help; transition: 0.3s;
    }
    .action-pill:hover { background: var(--primary); color: #022c22; transform: scale(1.05); }

    /* --- Result "Pop" --- */
    .result-reveal {
        animation: reveal 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    @keyframes reveal { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

</style>
""", unsafe_allow_html=True)

# 3. TOP NAVIGATION
st.markdown("""
<div style="padding: 30px 60px 10px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center;">
    <div>
        <h1 style="font-family: 'Syne'; color: white; margin: 0; font-size: 2.5rem;">Motion<span style="color:var(--primary)">IQ</span></h1>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">PRE-FINAL YEAR LAB EXHIBITION 2026</p>
    </div>
    <div style="text-align: right;">
        <span style="background: rgba(16,185,129,0.1); color: var(--primary); padding: 5px 15px; border-radius: 20px; font-size: 0.7rem; font-weight: bold; border: 1px solid var(--border);">SYSTEM ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 4. PRIMARY WORKSPACE
st.write("")
col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.markdown('<div style="padding-left: 60px;">', unsafe_allow_html=True)
    
    # Clickable Info Cards
    st.markdown("""
    <div class="action-grid">
        <div class="action-pill" title="Detects striking patterns">🥊 BOXING</div>
        <div class="action-pill" title="Monitors gait symmetry">🚶 WALKING</div>
        <div class="action-pill" title="Analyzes sprint mechanics">⚡ RUNNING</div>
        <div class="action-pill" title="Tracks rehab consistency">🏃 JOGGING</div>
        <div class="action-pill" title="Gesture interaction">👏 CLAPPING</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Source clip", type=["mp4", "webm"], label_visibility="collapsed")
    
    if uploaded:
        st.markdown('<div class="scanner-container">', unsafe_allow_html=True)
        if "analyzing" in st.session_state and st.session_state.analyzing:
            st.markdown('<div class="scanner-line"></div>', unsafe_allow_html=True)
        st.video(uploaded)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 INITIATE NEURAL INFERENCE", use_container_width=True):
            st.session_state.analyzing = True
            with st.spinner("Processing Spatiotemporal Tensors..."):
                time.sleep(2) # Simulate heavy computation
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div style="padding-right: 60px; background: rgba(30, 41, 59, 0.3); padding: 30px; border-radius: 15px; height: 100%;">', unsafe_allow_html=True)
    st.markdown("<h3 style='color:white; font-family:Syne;'>LIVE ANALYTICS</h3>", unsafe_allow_html=True)
    
    if "analyzing" not in st.session_state:
        st.info("Input stream required for activation.")
        st.write("---")
        st.caption("Awaiting 15fps sampling buffer...")
    else:
        # Result Wrapper with Animation
        st.markdown('<div class="result-reveal">', unsafe_allow_html=True)
        st.success("Target Identified")
        st.markdown("<h1 style='color:#10b981; margin:0; font-family:Syne;'>WALKING</h1>", unsafe_allow_html=True)
        
        # Engagement Gauge
        st.write("")
        st.progress(0.96, text="Neural Confidence Score: 96.4%")
        
        # Extra Metric Grid
        m1, m2 = st.columns(2)
        m1.metric("Frame Latency", "12ms", "-2ms")
        m2.metric("Gait Score", "0.98", "Optimal")
        
        st.code("DETECTED_ACTION: 05_WALK\nCOORD_ACC: 0.9921\nARCH: CONVLSTM_V2", language="bash")
        
        if st.button("RESET ENGINE"):
            del st.session_state["analyzing"]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 5. DYNAMIC FOOTER (Social Impact)
st.markdown("""
<div style="margin-top: 40px; padding: 60px; background: #0a1120; border-top: 1px solid var(--border);">
    <h2 style="color: white; font-family: 'Syne'; text-align: center;">Social Utility Integration</h2>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 30px; margin-top: 30px;">
        <div style="text-align: center;">
            <div style="font-size: 2rem;">🏥</div>
            <h4 style="color: var(--primary);">Clinical Rehab</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">Quantifying patient recovery metrics automatically.</p>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2rem;">👵</div>
            <h4 style="color: var(--primary);">Elder Care</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">Predicting falls via gait deviation analysis.</p>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2rem;">♿</div>
            <h4 style="color: var(--primary);">Assistive Tech</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">Touchless interfaces for motor-impaired users.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import streamlit.components.v1 as components
import time
import numpy as np

# ─────────────────────────────────────────────
# 1. PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MotionIQ | Neural Action Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# 2. GLOBAL STYLES + ANIMATIONS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;700;900&family=Share+Tech+Mono&display=swap');

:root {
    --neon:   #00ffe7;
    --neon2:  #ff2d78;
    --neon3:  #ffe600;
    --bg:     #080c14;
    --panel:  #0d1526;
    --glass:  rgba(0,255,231,0.06);
    --border: rgba(0,255,231,0.18);
    --text:   #cce8e4;
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
[data-testid="stAppViewContainer"] > section:first-child { padding: 0 !important; }
[data-testid="block-container"] { padding: 0 !important; max-width: 100% !important; }
.stButton > button { width: 100%; }

/* ── Grid bg ── */
body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(0,255,231,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,231,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* ── Particles canvas layer ── */
#particles-canvas {
    position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.35;
}

/* ── Main shell ── */
.app-shell { position: relative; z-index: 1; }

/* ── NAV ── */
.nav-bar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 22px 60px;
    border-bottom: 1px solid var(--border);
    background: rgba(8,12,20,0.85);
    backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
}
.logo { font-family: 'Exo 2'; font-weight: 900; font-size: 2.2rem; color: white; }
.logo span { color: var(--neon); text-shadow: 0 0 18px var(--neon); }
.nav-badge {
    display: flex; gap: 12px; align-items: center;
}
.badge-pulse {
    display: flex; align-items: center; gap: 8px;
    background: rgba(0,255,231,0.08);
    border: 1px solid var(--border);
    border-radius: 30px; padding: 6px 16px;
    font-family: 'Share Tech Mono'; font-size: 0.72rem; color: var(--neon);
}
.dot-live {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--neon);
    box-shadow: 0 0 8px var(--neon);
    animation: pulse-dot 1.2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { transform: scale(1); opacity: 1; }
    50%      { transform: scale(1.6); opacity: 0.5; }
}

/* ── Hero strip ── */
.hero-strip {
    text-align: center; padding: 48px 20px 20px;
}
.hero-strip h2 {
    font-family: 'Exo 2'; font-weight: 900; font-size: clamp(1.6rem,3vw,2.6rem);
    color: white; margin: 0 0 10px;
}
.hero-strip h2 em {
    font-style: normal;
    background: linear-gradient(90deg, var(--neon), var(--neon2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-strip p {
    font-family: 'Share Tech Mono'; color: #5a7a75; font-size: 0.82rem; margin: 0;
}

/* ── Tag pills ── */
.tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
.tag {
    background: var(--glass); border: 1px solid var(--border);
    border-radius: 30px; padding: 6px 14px;
    font-family: 'Share Tech Mono'; font-size: 0.68rem; color: var(--neon);
    cursor: default; transition: all 0.25s;
    user-select: none;
}
.tag:hover {
    background: var(--neon); color: #001a15;
    box-shadow: 0 0 18px var(--neon);
    transform: translateY(-2px);
}

/* ── Upload zone ── */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: 14px; padding: 30px 20px;
    text-align: center;
    background: var(--glass);
    transition: border-color 0.3s;
    margin-bottom: 16px;
}
[data-testid="stFileUploader"] {
    background: transparent !important;
}

/* ── Section headers ── */
.sec-header {
    font-family: 'Exo 2'; font-weight: 700; font-size: 1rem;
    color: var(--neon); letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 14px;
    display: flex; align-items: center; gap: 10px;
}
.sec-header::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(to right, var(--border), transparent);
}

/* ── Scanner video wrapper ── */
.scan-wrap { position: relative; border-radius: 12px; overflow: hidden; }
.scan-line {
    position: absolute; left: 0; width: 100%; height: 3px;
    background: linear-gradient(90deg, transparent, var(--neon), transparent);
    box-shadow: 0 0 20px var(--neon), 0 0 60px var(--neon);
    animation: scan-anim 2.4s ease-in-out infinite;
    z-index: 20; top: 0;
}
@keyframes scan-anim {
    0%   { top: 0%;   opacity: 0; }
    5%   { opacity: 1; }
    95%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}
.corner {
    position: absolute; width: 22px; height: 22px; z-index: 20;
    border-color: var(--neon); border-style: solid;
}
.corner.tl { top: 8px; left: 8px; border-width: 2px 0 0 2px; }
.corner.tr { top: 8px; right: 8px; border-width: 2px 2px 0 0; }
.corner.bl { bottom: 8px; left: 8px; border-width: 0 0 2px 2px; }
.corner.br { bottom: 8px; right: 8px; border-width: 0 2px 2px 0; }

/* ── Result panel ── */
.result-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 24px;
    animation: pop-in 0.6s cubic-bezier(0.175,0.885,0.32,1.275);
}
@keyframes pop-in {
    from { opacity: 0; transform: scale(0.92) translateY(16px); }
    to   { opacity: 1; transform: scale(1)    translateY(0); }
}
.action-name {
    font-family: 'Exo 2'; font-weight: 900;
    font-size: clamp(2rem,4vw,3.2rem);
    color: var(--neon);
    text-shadow: 0 0 30px var(--neon), 0 0 60px rgba(0,255,231,0.3);
    margin: 8px 0;
}
.conf-bar-wrap {
    background: rgba(0,255,231,0.07);
    border-radius: 6px; overflow: hidden;
    height: 10px; margin: 10px 0;
    border: 1px solid var(--border);
}
.conf-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--neon), #00b894);
    box-shadow: 0 0 14px var(--neon);
    border-radius: 6px;
    animation: bar-fill 1.2s ease forwards;
}
@keyframes bar-fill { from { width: 0; } }
.metric-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }
.metric-card {
    background: var(--glass); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
    font-family: 'Share Tech Mono';
}
.metric-card .val {
    font-size: 1.5rem; color: var(--neon); font-weight: bold;
    text-shadow: 0 0 10px var(--neon);
}
.metric-card .lbl { font-size: 0.65rem; color: #4a7a74; margin-top: 2px; }
.metric-card .delta { font-size: 0.7rem; color: #00ffe7; opacity: 0.7; }

/* ── Live record panel ── */
.live-panel {
    background: var(--panel);
    border: 1px solid rgba(255,45,120,0.3);
    border-radius: 16px;
    padding: 24px;
    margin-top: 10px;
}
.live-title {
    font-family: 'Exo 2'; font-weight: 700; font-size: 1rem;
    color: var(--neon2); letter-spacing: 2px;
    display: flex; align-items: center; gap: 8px; margin-bottom: 14px;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Exo 2' !important; font-weight: 700 !important;
    letter-spacing: 2px !important; border-radius: 8px !important;
    border: none !important; cursor: pointer !important;
    transition: all 0.2s !important;
}
button[kind="primary"],
.stButton > button:not([kind]) {
    background: linear-gradient(135deg, var(--neon), #00b894) !important;
    color: #001a15 !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(0,255,231,0.35) !important;
}

/* ── Footer ── */
.footer-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 24px; margin-top: 24px;
}
.foot-card {
    background: var(--glass); border: 1px solid var(--border);
    border-radius: 14px; padding: 24px 20px; text-align: center;
    transition: transform 0.3s, box-shadow 0.3s;
}
.foot-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 40px rgba(0,255,231,0.1);
}
.foot-icon { font-size: 2rem; margin-bottom: 10px; }
.foot-title { font-family: 'Exo 2'; font-weight: 700; color: var(--neon); font-size: 0.95rem; }
.foot-body  { font-family: 'Share Tech Mono'; color: #5a7a75; font-size: 0.75rem; margin-top: 6px; }

/* ── Stick figure canvas ── */
#stick-canvas {
    display: block; margin: 0 auto;
    filter: drop-shadow(0 0 8px var(--neon));
}

/* ── Tab-like mode switch ── */
.mode-switch {
    display: flex; gap: 0; margin-bottom: 20px;
    border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
}
.mode-btn {
    flex: 1; padding: 10px; text-align: center;
    font-family: 'Share Tech Mono'; font-size: 0.75rem;
    cursor: pointer; transition: all 0.2s;
    color: #5a7a75; background: transparent;
}
.mode-btn.active {
    background: var(--neon); color: #001a15; font-weight: bold;
}

/* ── Streamlit overrides ── */
[data-testid="stVideo"] { border-radius: 10px; overflow: hidden; }
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--neon), #00b894) !important;
}
[data-testid="stCodeBlock"] {
    background: #030609 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
div[data-testid="metric-container"] {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. PARTICLE + STICK FIGURE JS (injected once)
# ─────────────────────────────────────────────
CANVAS_JS = """
<canvas id="particles-canvas"></canvas>
<script>
(function(){
  const c = document.getElementById('particles-canvas');
  const ctx = c.getContext('2d');
  function resize(){ c.width=window.innerWidth; c.height=window.innerHeight; }
  resize(); window.addEventListener('resize', resize);

  const pts = Array.from({length:55}, ()=>({
    x: Math.random()*window.innerWidth,
    y: Math.random()*window.innerHeight,
    vx:(Math.random()-.5)*.4,
    vy:(Math.random()-.5)*.4,
    r: Math.random()*1.5+.5
  }));

  function drawParticles(){
    ctx.clearRect(0,0,c.width,c.height);
    ctx.fillStyle='rgba(0,255,231,0.7)';
    pts.forEach(p=>{
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0||p.x>c.width)  p.vx*=-1;
      if(p.y<0||p.y>c.height) p.vy*=-1;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2); ctx.fill();
    });
    // lines
    ctx.strokeStyle='rgba(0,255,231,0.07)';
    ctx.lineWidth=.8;
    for(let i=0;i<pts.length;i++)
      for(let j=i+1;j<pts.length;j++){
        const dx=pts[i].x-pts[j].x, dy=pts[i].y-pts[j].y;
        if(Math.sqrt(dx*dx+dy*dy)<120){
          ctx.beginPath(); ctx.moveTo(pts[i].x,pts[i].y);
          ctx.lineTo(pts[j].x,pts[j].y); ctx.stroke();
        }
      }
    requestAnimationFrame(drawParticles);
  }
  drawParticles();
})();
</script>
"""

STICK_JS = """
<canvas id="stick-canvas" width="340" height="240"></canvas>
<script>
(function(){
  const cv = document.getElementById('stick-canvas');
  const ctx = cv.getContext('2d');
  const NEO = '#00ffe7';
  const NEO2 = '#ff2d78';
  let t = 0;

  // Skeleton joints: head, neck, lShoulder, rShoulder, lElbow, rElbow,
  //                  lWrist, rWrist, lHip, rHip, lKnee, rKnee, lFoot, rFoot
  function buildPose(anim, phase){
    const cx=170, cy=60;
    if(anim==='walk'){
      const leg = Math.sin(phase)*28;
      const arm = -Math.sin(phase)*24;
      return {
        head:[cx,cy], neck:[cx,cy+22],
        ls:[cx-20,cy+28], rs:[cx+20,cy+28],
        le:[cx-30+arm,cy+52], re:[cx+30-arm,cy+52],
        lw:[cx-38+arm,cy+72], rw:[cx+38-arm,cy+72],
        lh:[cx-12,cy+66], rh:[cx+12,cy+66],
        lk:[cx-14+leg*.4,cy+96], rk:[cx+14-leg*.4,cy+96],
        lf:[cx-16+leg,cy+124], rf:[cx+16-leg,cy+124]
      };
    }
    if(anim==='run'){
      const leg = Math.sin(phase)*42;
      const arm = -Math.sin(phase)*34;
      return {
        head:[cx,cy], neck:[cx,cy+22],
        ls:[cx-20,cy+28], rs:[cx+20,cy+28],
        le:[cx-28+arm,cy+52], re:[cx+28-arm,cy+52],
        lw:[cx-40+arm,cy+68], rw:[cx+40-arm,cy+68],
        lh:[cx-12,cy+66], rh:[cx+12,cy+66],
        lk:[cx-18+leg*.5,cy+92], rk:[cx+18-leg*.5,cy+92],
        lf:[cx-20+leg,cy+126], rf:[cx+20-leg,cy+126]
      };
    }
    if(anim==='boxing'){
      const punch = Math.max(0,Math.sin(phase))*40;
      const guard = 10;
      return {
        head:[cx,cy], neck:[cx,cy+22],
        ls:[cx-20,cy+28], rs:[cx+20,cy+28],
        le:[cx-26,cy+50], re:[cx+26-punch*.3,cy+46],
        lw:[cx-32+guard,cy+40], rw:[cx+52+punch,cy+36],
        lh:[cx-12,cy+68], rh:[cx+12,cy+68],
        lk:[cx-18,cy+98], rk:[cx+18+6,cy+96],
        lf:[cx-20,cy+128], rf:[cx+22,cy+128]
      };
    }
    if(anim==='clap'){
      const cl = Math.abs(Math.sin(phase))*40;
      return {
        head:[cx,cy], neck:[cx,cy+22],
        ls:[cx-20,cy+28], rs:[cx+20,cy+28],
        le:[cx-18+cl*.5,cy+54], re:[cx+18-cl*.5,cy+54],
        lw:[cx-8+cl,cy+76], rw:[cx+8-cl,cy+76],
        lh:[cx-12,cy+68], rh:[cx+12,cy+68],
        lk:[cx-14,cy+100], rk:[cx+14,cy+100],
        lf:[cx-16,cy+130], rf:[cx+16,cy+130]
      };
    }
    // default: idle
    const sw = Math.sin(phase*.5)*8;
    return {
      head:[cx,cy], neck:[cx,cy+22],
      ls:[cx-20,cy+28], rs:[cx+20,cy+28],
      le:[cx-28+sw,cy+54], re:[cx+28-sw,cy+54],
      lw:[cx-32+sw,cy+76], rw:[cx+32-sw,cy+76],
      lh:[cx-12,cy+68], rh:[cx+12,cy+68],
      lk:[cx-14,cy+100], rk:[cx+14,cy+100],
      lf:[cx-16,cy+130], rf:[cx+16,cy+130]
    };
  }

  function drawBone(p1,p2,glow,w){
    ctx.beginPath(); ctx.moveTo(...p1); ctx.lineTo(...p2);
    ctx.strokeStyle=glow; ctx.lineWidth=w||3;
    ctx.lineCap='round';
    ctx.shadowColor=glow; ctx.shadowBlur=12;
    ctx.stroke();
  }
  function drawJoint(p,glow,r){
    ctx.beginPath(); ctx.arc(p[0],p[1],r||5,0,Math.PI*2);
    ctx.fillStyle=glow; ctx.shadowColor=glow; ctx.shadowBlur=16;
    ctx.fill();
  }

  // Read current animation from a data attribute we'll set from Python-rendered HTML
  function getAnim(){
    const el=document.getElementById('stick-anim-state');
    return el ? el.dataset.anim : 'idle';
  }

  function draw(){
    ctx.clearRect(0,0,cv.width,cv.height);
    const anim=getAnim();
    const speed = (anim==='run') ? 0.12 : (anim==='boxing') ? 0.10 : 0.07;
    t+=speed;
    const P=buildPose(anim,t);

    ctx.shadowBlur=0;
    // bones
    const boneColor='#00ffe7'; const accentColor='#ff2d78';
    drawBone(P.neck,P.head,boneColor,3);
    drawBone(P.neck,P.ls,boneColor,3);
    drawBone(P.neck,P.rs,boneColor,3);
    drawBone(P.ls,P.le,boneColor,3);
    drawBone(P.rs,P.re,boneColor,3);
    drawBone(P.le,P.lw,accentColor,2.5);
    drawBone(P.re,P.rw,accentColor,2.5);
    drawBone(P.neck,P.lh,boneColor,4);
    drawBone(P.neck,P.rh,boneColor,4);
    drawBone(P.lh,P.lk,boneColor,3.5);
    drawBone(P.rh,P.rk,boneColor,3.5);
    drawBone(P.lk,P.lf,accentColor,3);
    drawBone(P.rk,P.rf,accentColor,3);
    // joints
    [P.head,P.neck,P.ls,P.rs,P.lh,P.rh].forEach(p=>drawJoint(p,boneColor,5));
    [P.le,P.re,P.lk,P.rk].forEach(p=>drawJoint(p,boneColor,4));
    [P.lw,P.rw,P.lf,P.rf].forEach(p=>drawJoint(p,accentColor,3.5));
    // head circle
    drawJoint(P.head,'#00ffe7',12);

    // ground line
    ctx.beginPath(); ctx.moveTo(60,cv.height-10); ctx.lineTo(280,cv.height-10);
    ctx.strokeStyle='rgba(0,255,231,0.18)'; ctx.lineWidth=1;
    ctx.shadowBlur=0; ctx.stroke();

    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
"""

components.html(CANVAS_JS, height=0)

# ─────────────────────────────────────────────
# 4. NAV BAR
# ─────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
  <div class="logo">Motion<span>IQ</span></div>
  <div class="nav-badge">
    <div class="badge-pulse"><div class="dot-live"></div>NEURAL ENGINE ONLINE</div>
    <div class="badge-pulse" style="border-color:rgba(255,45,120,0.3);color:#ff2d78;">KTH · 6-CLASS MODEL</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 5. HERO STRIP
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-strip">
  <h2>Real-Time <em>Human Action Recognition</em></h2>
  <p>ConvLSTM · Spatiotemporal Tensor Analysis · Sub-20ms Latency · Pre-Final Year Lab Exhibition 2026</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 6. SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
if "result"     not in st.session_state: st.session_state.result     = None
if "mode"       not in st.session_state: st.session_state.mode       = "upload"
if "analyzing"  not in st.session_state: st.session_state.analyzing  = False
if "anim_state" not in st.session_state: st.session_state.anim_state = "idle"

ACTION_META = {
    "WALKING":   {"icon":"🚶","score":0.964,"latency":"11ms","extra":"Gait Score: 0.98",   "anim":"walk"},
    "RUNNING":   {"icon":"⚡","score":0.951,"latency":"13ms","extra":"Cadence: 172 spm",    "anim":"run"},
    "JOGGING":   {"icon":"🏃","score":0.938,"latency":"14ms","extra":"Stride Sym: 0.96",   "anim":"run"},
    "BOXING":    {"icon":"🥊","score":0.977,"latency":"9ms", "extra":"Punch Vel: High",    "anim":"boxing"},
    "CLAPPING":  {"icon":"👏","score":0.922,"latency":"10ms","extra":"Freq: 2.4 Hz",       "anim":"clap"},
    "HANDWAVING":{"icon":"🖐️","score":0.911,"latency":"12ms","extra":"Amp: 0.87",          "anim":"idle"},
}

# ─────────────────────────────────────────────
# 7. MAIN LAYOUT
# ─────────────────────────────────────────────
st.write("")
left, right = st.columns([1.15, 0.85], gap="large")

# ── LEFT COLUMN ──────────────────────────────
with left:
    st.markdown('<div style="padding-left:50px;padding-right:20px;">', unsafe_allow_html=True)

    # Action tags
    st.markdown("""
    <div class="tag-row">
      <div class="tag">🥊 BOXING</div>
      <div class="tag">🚶 WALKING</div>
      <div class="tag">⚡ RUNNING</div>
      <div class="tag">🏃 JOGGING</div>
      <div class="tag">👏 CLAPPING</div>
      <div class="tag">🖐️ HANDWAVING</div>
    </div>
    """, unsafe_allow_html=True)

    # Mode switcher
    st.markdown('<div class="sec-header">INPUT SOURCE</div>', unsafe_allow_html=True)
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        if st.button("📁 Upload Video", use_container_width=True,
                     type="primary" if st.session_state.mode=="upload" else "secondary"):
            st.session_state.mode = "upload"
            st.session_state.result = None
            st.session_state.anim_state = "idle"
            st.rerun()
    with mode_col2:
        if st.button("🔴 Live Webcam", use_container_width=True,
                     type="primary" if st.session_state.mode=="live" else "secondary"):
            st.session_state.mode = "live"
            st.session_state.result = None
            st.rerun()

    st.write("")

    # ── UPLOAD MODE ──
    if st.session_state.mode == "upload":
        uploaded = st.file_uploader(
            "Drop your video clip here",
            type=["mp4","webm","avi","mov"],
            label_visibility="visible"
        )

        if uploaded:
            st.markdown('<div class="scan-wrap">', unsafe_allow_html=True)
            if st.session_state.analyzing:
                st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="corner tl"></div><div class="corner tr"></div>
            <div class="corner bl"></div><div class="corner br"></div>
            """, unsafe_allow_html=True)
            st.video(uploaded)
            st.markdown('</div>', unsafe_allow_html=True)

            st.write("")
            if st.button("⚡ INITIATE NEURAL INFERENCE", use_container_width=True, type="primary"):
                st.session_state.analyzing = True
                with st.spinner("Processing spatiotemporal tensors..."):
                    time.sleep(2.2)

                # ── Simulate model result ──
                # In production: replace this block with your real model call
                actions = list(ACTION_META.keys())
                chosen = actions[np.random.randint(len(actions))]
                st.session_state.result = chosen
                st.session_state.anim_state = ACTION_META[chosen]["anim"]
                st.session_state.analyzing = False
                st.rerun()

    # ── LIVE MODE ──
    else:
        st.markdown('<div class="live-panel">', unsafe_allow_html=True)
        st.markdown('<div class="live-title"><div class="dot-live" style="background:#ff2d78;box-shadow:0 0 8px #ff2d78;"></div> LIVE WEBCAM CAPTURE</div>', unsafe_allow_html=True)

        cam_img = st.camera_input("Point your camera and capture a frame")

        if cam_img:
            st.success("Frame captured! Running inference...")
            with st.spinner("Analysing frame via ConvLSTM..."):
                time.sleep(1.8)

            # Simulate model result for captured frame
            actions = list(ACTION_META.keys())
            chosen = actions[np.random.randint(len(actions))]
            st.session_state.result = chosen
            st.session_state.anim_state = ACTION_META[chosen]["anim"]
            st.rerun()

        st.markdown("""
        <p style="font-family:'Share Tech Mono';color:#5a7a75;font-size:0.72rem;margin-top:12px;">
        ⚠ Webcam access uses browser permissions. Frame is processed locally and not stored.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.result:
            st.write("")
            if st.button("🔄 CAPTURE NEW FRAME", use_container_width=True):
                st.session_state.result = None
                st.session_state.anim_state = "idle"
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT COLUMN ─────────────────────────────
with right:
    st.markdown('<div style="padding-right:50px;padding-left:10px;">', unsafe_allow_html=True)
    st.markdown('<div class="sec-header">ANIMATED SKELETON</div>', unsafe_allow_html=True)

    # Stick figure canvas + anim state bridge
    anim_val = st.session_state.anim_state
    stick_html = f"""
    <span id="stick-anim-state" data-anim="{anim_val}" style="display:none;"></span>
    {STICK_JS}
    """
    components.html(stick_html, height=260)

    st.write("")
    st.markdown('<div class="sec-header">LIVE ANALYTICS</div>', unsafe_allow_html=True)

    if st.session_state.result is None:
        st.markdown("""
        <div style="background:var(--glass);border:1px solid var(--border);border-radius:14px;
                    padding:30px 24px;text-align:center;">
          <div style="font-size:2.8rem;margin-bottom:12px;">🎯</div>
          <div style="font-family:'Share Tech Mono';color:#5a7a75;font-size:0.8rem;">
            AWAITING INPUT STREAM<br><br>
            Upload a video clip or use live webcam<br>
            to activate neural inference pipeline.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        res = st.session_state.result
        meta = ACTION_META[res]
        conf_pct = int(meta["score"]*100)
        conf_w   = f"{conf_pct}%"

        st.markdown(f"""
        <div class="result-panel">
          <div style="font-family:'Share Tech Mono';color:#5a7a75;font-size:0.7rem;letter-spacing:2px;">
            DETECTED ACTION
          </div>
          <div class="action-name">{meta['icon']} {res}</div>

          <div style="font-family:'Share Tech Mono';color:#5a7a75;font-size:0.68rem;margin-bottom:4px;">
            NEURAL CONFIDENCE · {conf_pct}%
          </div>
          <div class="conf-bar-wrap">
            <div class="conf-bar" style="width:{conf_w};"></div>
          </div>

          <div class="metric-row">
            <div class="metric-card">
              <div class="val">{meta['latency']}</div>
              <div class="lbl">FRAME LATENCY</div>
              <div class="delta">↓ optimal</div>
            </div>
            <div class="metric-card">
              <div class="val">{conf_pct/100:.3f}</div>
              <div class="lbl">COORD ACC</div>
              <div class="delta">↑ high</div>
            </div>
            <div class="metric-card">
              <div class="val">V2</div>
              <div class="lbl">ARCH · CONVLSTM</div>
              <div class="delta">KTH DATASET</div>
            </div>
            <div class="metric-card">
              <div class="val" style="font-size:1rem;">{meta['extra']}</div>
              <div class="lbl">MOTION METRIC</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.code(
            f"DETECTED_ACTION : {res}\n"
            f"CONFIDENCE      : {meta['score']:.4f}\n"
            f"FRAME_LATENCY   : {meta['latency']}\n"
            f"ARCH            : CONVLSTM_V2\n"
            f"DATASET         : KTH_6CLASS",
            language="bash"
        )

        if st.button("🔄 RESET ENGINE", use_container_width=True):
            st.session_state.result     = None
            st.session_state.analyzing  = False
            st.session_state.anim_state = "idle"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 8. FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-top:60px;padding:60px;background:#050810;border-top:1px solid var(--border);">
  <div style="text-align:center;margin-bottom:32px;">
    <div style="font-family:'Exo 2';font-weight:900;font-size:1.6rem;color:white;">
      Social <span style="color:var(--neon);text-shadow:0 0 20px var(--neon);">Utility</span> Integration
    </div>
    <div style="font-family:'Share Tech Mono';color:#5a7a75;font-size:0.75rem;margin-top:6px;">
      REAL-WORLD DEPLOYMENT SCENARIOS
    </div>
  </div>
  <div class="footer-grid">
    <div class="foot-card">
      <div class="foot-icon">🏥</div>
      <div class="foot-title">CLINICAL REHAB</div>
      <div class="foot-body">Automatically quantifies patient recovery metrics and gait symmetry post-surgery.</div>
    </div>
    <div class="foot-card">
      <div class="foot-icon">👵</div>
      <div class="foot-title">ELDER CARE</div>
      <div class="foot-body">Predicts fall risk via real-time gait deviation analysis with silent alerting.</div>
    </div>
    <div class="foot-card">
      <div class="foot-icon">♿</div>
      <div class="foot-title">ASSISTIVE TECH</div>
      <div class="foot-body">Enables touchless gesture interfaces for users with motor impairments.</div>
    </div>
  </div>
  <div style="text-align:center;margin-top:40px;font-family:'Share Tech Mono';color:#2a3a38;font-size:0.7rem;">
    MotionIQ · Pre-Final Year Lab Exhibition 2026 · Built with ConvLSTM on KTH Dataset
  </div>
</div>
""", unsafe_allow_html=True)

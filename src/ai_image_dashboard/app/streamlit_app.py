"""
CreatorAI — Unified AI Image Processing Studio
"""
import streamlit as st
import os
import sys
import time
import numpy as np
import cv2
import torch

# ── Path Setup ──────────────────────────────────────────────────
DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(DASHBOARD_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)

from utils.image_utils import bytes_to_rgb, rgb_to_png_bytes, resize_safe
from app.config import TOOLS

# ═══════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CreatorAI Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# Premium CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Reset & Base ── */
    .stApp {
        font-family: 'Inter', -apple-system, sans-serif;
        background: #0A0A0F;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D0D14 0%, #111118 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.04) !important;
        width: 300px !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem !important;
    }

    /* ── Sidebar Brand ── */
    .sidebar-brand {
        text-align: center;
        padding: 1.2rem 1rem 1rem;
        margin-bottom: 0.5rem;
    }
    .sidebar-brand h2 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #A78BFA, #7C3AED, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-brand p {
        margin: 0.3rem 0 0;
        font-size: 0.7rem;
        color: #4B5563;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 500;
    }

    /* ── Sidebar Dividers ── */
    .sidebar-section-label {
        font-size: 0.65rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        padding: 0.8rem 0 0.4rem;
        margin-top: 0.5rem;
    }

    .sidebar-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.2), transparent);
        margin: 0.6rem 0;
    }

    /* ── Tool Buttons (sidebar) ── */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        color: #9CA3AF !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.7rem 1rem !important;
        text-align: left !important;
        transition: all 0.25s ease !important;
        margin-bottom: 0.15rem !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(139, 92, 246, 0.08) !important;
        border-color: rgba(139, 92, 246, 0.25) !important;
        color: #E5E7EB !important;
        transform: translateX(3px) !important;
    }

    /* ── Active tool highlight ── */
    .active-tool-btn > button {
        background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(236,72,153,0.08)) !important;
        border-color: rgba(139, 92, 246, 0.4) !important;
        color: #E5E7EB !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.1) !important;
    }

    /* ── Main content buttons ── */
    .main-action > button {
        background: linear-gradient(135deg, #7C3AED, #A78BFA) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .main-action > button:hover {
        box-shadow: 0 6px 30px rgba(124, 58, 237, 0.45) !important;
        transform: translateY(-2px) !important;
    }

    .secondary-btn > button {
        background: rgba(255,255,255,0.05) !important;
        color: #9CA3AF !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #0F0A1A 0%, #1A1025 50%, #0F0A1A 100%);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #C4B5FD, #A78BFA, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
    }
    .main-header .subtitle {
        color: #6B7280;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        font-weight: 300;
        position: relative;
    }

    /* ── Status Chip ── */
    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 100px;
        padding: 0.45rem 1rem;
        margin: 1rem 0 0.5rem;
        font-size: 0.78rem;
        color: #9CA3AF;
    }
    .status-chip .dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
    }
    .status-chip .dot.cpu { background: #F59E0B; box-shadow: 0 0 8px rgba(245, 158, 11, 0.5); }

    /* ── Card Container ── */
    .feature-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .feature-card:hover {
        border-color: rgba(139, 92, 246, 0.2);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }
    .feature-card .icon { font-size: 2rem; margin-bottom: 0.6rem; }
    .feature-card .name {
        font-weight: 700;
        font-size: 0.95rem;
        color: #E5E7EB;
        margin-bottom: 0.3rem;
    }
    .feature-card .desc {
        font-size: 0.72rem;
        color: #6B7280;
        line-height: 1.4;
    }

    /* ── Image Frame ── */
    .img-frame {
        background: #0D0D12;
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 14px;
        padding: 0.6rem;
        margin: 0.5rem 0;
    }
    .img-label {
        font-size: 0.7rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }



    /* ── Metric boxes ── */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        margin: 0.8rem 0;
    }
    .metric-box {
        flex: 1;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
    }
    .metric-box .value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #A78BFA;
    }
    .metric-box .label {
        font-size: 0.65rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
    }

    /* ── Processing animation ── */
    .proc-text {
        color: #A78BFA;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* ── Hide Streamlit chrome (preserve sidebar toggle) ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    /* Keep header visible so sidebar toggle arrow works */
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
    }

    /* ── Spinner override ── */
    .stSpinner > div {
        border-top-color: #7C3AED !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════════
if "current_tool" not in st.session_state:
    st.session_state.current_tool = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "history" not in st.session_state:
    st.session_state.history = []
if "process_time" not in st.session_state:
    st.session_state.process_time = 0.0


# ═══════════════════════════════════════════════════════════════
# Cached Model Loaders
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_super_resolution():
    from pipelines import super_resolution_pipeline
    return super_resolution_pipeline.get_pipeline()

@st.cache_resource
def load_text_removal():
    from pipelines import text_removal_pipeline
    return text_removal_pipeline.get_pipeline()

@st.cache_resource
def load_color_correction():
    from pipelines import color_pipeline
    return color_pipeline.get_pipeline()

@st.cache_resource
def load_object_removal():
    from pipelines import object_pipeline
    return object_pipeline.get_pipeline()

@st.cache_resource
def load_background_removal():
    from pipelines import background_pipeline
    return background_pipeline.get_pipeline()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Brand ──
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🚀 CreatorAI</h2>
        <p>Image Studio</p>
    </div>
    <hr class="sidebar-divider">
    """, unsafe_allow_html=True)

    # ── GPU Status ──
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        dot_class = "dot"
        status_text = f"{gpu_name} • {vram_gb:.1f}GB"
    else:
        dot_class = "dot cpu"
        status_text = "CPU Mode"

    st.markdown(f"""
    <div class="status-chip">
        <div class="{dot_class}"></div>
        <span>{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")  # spacer

    # ── Tool Selection ──
    st.markdown('<div class="sidebar-section-label">🎯 AI Tools</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    for tool_key, tool_info in TOOLS.items():
        is_active = st.session_state.current_tool == tool_key
        container_class = "active-tool-btn" if is_active else ""

        with st.container():
            if is_active:
                st.markdown(f'<div class="active-tool-btn">', unsafe_allow_html=True)
            if st.button(
                f"{tool_info['icon']}   {tool_info['name']}",
                key=f"tool_{tool_key}",
                width='stretch',
            ):
                st.session_state.current_tool = tool_key
                st.session_state.processed_image = None
                st.session_state.process_time = 0.0
                st.rerun()
            if is_active:
                st.markdown('</div>', unsafe_allow_html=True)



    # ── History ──
    if st.session_state.history:
        st.markdown("")
        st.markdown("")
        st.markdown('<div class="sidebar-section-label">📜 Recent</div>', unsafe_allow_html=True)
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        for entry in reversed(st.session_state.history[-4:]):
            st.markdown(f"&nbsp; {entry['tool_icon']} {entry['tool_name']} — *{entry['time']}*")

        st.markdown("")
        if st.button("↩️  Undo Last", width='stretch', key="undo_btn"):
            if st.session_state.history:
                last = st.session_state.history.pop()
                st.session_state.processed_image = last.get("prev_image")
                st.rerun()


# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════

if st.session_state.current_tool is None:
    # ──────────────────────────────────────────────────────────
    # Landing Page
    # ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🚀 CreatorAI Studio</h1>
        <div class="subtitle">
            Production-grade AI image processing. Select a tool from the sidebar to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(5, gap="medium")
    for i, (key, info) in enumerate(TOOLS.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="icon">{info['icon']}</div>
                <div class="name">{info['name']}</div>
                <div class="desc">{info['description']}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ──────────────────────────────────────────────────────────
    # Tool Processing Page
    # ──────────────────────────────────────────────────────────
    tool = TOOLS[st.session_state.current_tool]

    # ── Mini Header ──
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:1.2rem;">
        <span style="font-size:1.8rem;">{tool['icon']}</span>
        <div>
            <div style="font-size:1.3rem; font-weight:700; color:#E5E7EB;">{tool['name']}</div>
            <div style="font-size:0.8rem; color:#6B7280;">{tool['description']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Default Tool Settings ──
    sr_scale = 4
    sr_mode = "fast"
    cc_restormer = True
    or_prompt = ""
    bg_mode = "auto"
    bg_show_metrics = False

    # ── Tool-Specific Settings UI ──
    if st.session_state.current_tool in ["super_resolution", "color_correction", "object_removal", "background_removal"]:
        with st.container(border=True):
            st.markdown("##### ⚙️ Settings")
            if st.session_state.current_tool == "super_resolution":
                sc_col1, sc_col2 = st.columns(2)
                with sc_col1:
                    sr_scale = st.select_slider("Upscale Factor", options=[2, 4], value=4)
                with sc_col2:
                    sr_mode = st.selectbox(
                        "🧠 Processing Mode",
                        options=["fast", "balanced", "hd"],
                        index=0,
                        format_func=lambda x: {"fast": "⚡ Fast (~15s)", "balanced": "⚖️ Balanced (~45s)", "hd": "🎨 HD (~3min)"}[x],
                        help="Fast: RealESRGAN + auto face restore | Balanced: + SwinIR if needed | HD: + Diffusion for extreme detail",
                    )
            elif st.session_state.current_tool == "color_correction":
                cc_restormer = st.toggle("✨ Restormer Refinement", value=True, help="Sharper details, less noise — uses more VRAM")
            elif st.session_state.current_tool == "object_removal":
                or_prompt = st.text_input("🎯 Describe the object", placeholder="person, car, sign, etc.", help="What should CreatorAI remove from the image?")
            elif st.session_state.current_tool == "background_removal":
                bg_col1, bg_col2 = st.columns(2)
                with bg_col1:
                    bg_mode = st.selectbox(
                        "🧠 Pipeline Mode",
                        options=["auto", "simple", "complex"],
                        index=0,
                        help="Auto: adapts per scene | Simple: portraits (fast) | Complex: products (precise)",
                    )
                with bg_col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    bg_show_metrics = st.toggle("📊 Show Latency Metrics", value=True, help="Display per-stage timing breakdown")
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload ──
    uploaded = st.file_uploader(
        "Drop your image here",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="uploader",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        image_bytes = uploaded.read()
        img_rgb = bytes_to_rgb(image_bytes)
        st.session_state.original_image = img_rgb

        # ── Action Bar ──
        act_col1, act_col2, act_col3, act_col4 = st.columns([3, 1.2, 1.2, 1.5])

        with act_col1:
            st.markdown('<div class="main-action">', unsafe_allow_html=True)
            process_btn = st.button(
                f"⚡  Process with {tool['name']}",
                width='stretch',
                key="process_btn",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with act_col2:
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            reset_btn = st.button("🔄 Reset", width='stretch', key="reset_btn")
            st.markdown('</div>', unsafe_allow_html=True)

        with act_col3:
            if st.session_state.processed_image is not None:
                processed = st.session_state.processed_image
                # Handle RGBA (background removal)
                if len(processed.shape) == 3 and processed.shape[2] == 4:
                    # Encode as PNG with alpha
                    success, buf = cv2.imencode(".png", cv2.cvtColor(processed, cv2.COLOR_RGBA2BGRA))
                    dl_bytes = buf.tobytes() if success else b""
                else:
                    dl_bytes = rgb_to_png_bytes(processed)
                st.download_button(
                    "📥 Download",
                    data=dl_bytes,
                    file_name=f"creatorai_{st.session_state.current_tool}.png",
                    mime="image/png",
                    width='stretch',
                    key="dl_btn",
                )

        with act_col4:
            if st.session_state.process_time > 0:
                st.markdown(f"""
                <div style="text-align:center; padding:0.5rem;">
                    <span style="color:#10B981; font-weight:700; font-size:1.1rem;">
                        {st.session_state.process_time:.1f}s
                    </span>
                    <br>
                    <span style="color:#6B7280; font-size:0.65rem; text-transform:uppercase; letter-spacing:1px;">
                        Process Time
                    </span>
                </div>
                """, unsafe_allow_html=True)

        # ── Reset Logic ──
        if reset_btn:
            st.session_state.processed_image = None
            st.session_state.process_time = 0.0
            st.session_state.last_processed_tool = None
            st.rerun()

        # ── Processing Logic ──
        if process_btn:
            with st.spinner(f"🔮 CreatorAI is processing your image..."):
                start = time.time()
                prev_image = st.session_state.processed_image
                last_tool = st.session_state.get("last_processed_tool", None)
                
                # Chain: if already processed BY A DIFFERENT TOOL, use that as input
                if prev_image is not None and last_tool != st.session_state.current_tool:
                    input_img = prev_image
                else:
                    input_img = img_rgb

                # Handle RGBA → RGB for chaining (if previous was bg removal)
                if len(input_img.shape) == 3 and input_img.shape[2] == 4:
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGBA2RGB)

                try:
                    result = None

                    if st.session_state.current_tool == "super_resolution":
                        pipeline = load_super_resolution()
                        from pipelines import super_resolution_pipeline
                        result = super_resolution_pipeline.process(
                            pipeline, input_img,
                            scale=sr_scale, mode=sr_mode
                        )

                    elif st.session_state.current_tool == "text_removal":
                        pipeline = load_text_removal()
                        from pipelines import text_removal_pipeline
                        result = text_removal_pipeline.process(pipeline, input_img)

                    elif st.session_state.current_tool == "color_correction":
                        pipeline = load_color_correction()
                        from pipelines import color_pipeline
                        result = color_pipeline.process(
                            pipeline, input_img, use_restormer=cc_restormer
                        )

                    elif st.session_state.current_tool == "object_removal":
                        if not or_prompt:
                            st.warning("⚠️ Please describe the object to remove in the sidebar.")
                        else:
                            pipeline = load_object_removal()
                            from pipelines import object_pipeline
                            result = object_pipeline.process(pipeline, input_img, prompt=or_prompt)

                    elif st.session_state.current_tool == "background_removal":
                        pipeline = load_background_removal()
                        from pipelines import background_pipeline
                        result, bg_metrics = background_pipeline.process(
                            pipeline, input_img, mode=bg_mode, return_metrics=True
                        )
                        st.session_state.bg_metrics = bg_metrics
                        st.session_state.bg_scene_type = bg_metrics.get("scene_type", "")

                    elapsed = time.time() - start

                    if result is not None:
                        st.session_state.processed_image = result
                        st.session_state.process_time = elapsed
                        st.session_state.last_processed_tool = st.session_state.current_tool
                        st.session_state.history.append({
                            "tool_name": tool["name"],
                            "tool_icon": tool["icon"],
                            "time": f"{elapsed:.1f}s",
                            "prev_image": prev_image,
                        })
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        st.exception(e)

        # ── Image Display ──
        st.markdown("---")

        if st.session_state.processed_image is not None:
            # ── Before / After ──
            col1, col2 = st.columns(2, gap="medium")

            with col1:
                st.markdown('<div class="img-label">📷 Original</div>', unsafe_allow_html=True)
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(img_rgb, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
                oh, ow = img_rgb.shape[:2]
                st.caption(f"{ow} × {oh} px")

            with col2:
                st.markdown('<div class="img-label">✨ Processed</div>', unsafe_allow_html=True)
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                processed = st.session_state.processed_image
                st.image(processed, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
                ph, pw = processed.shape[:2]
                ch = "RGBA" if (len(processed.shape) == 3 and processed.shape[2] == 4) else "RGB"
                st.caption(f"{pw} × {ph} px • {ch}")

            # ── Background Removal Metrics ──
            if (st.session_state.current_tool == "background_removal"
                    and bg_show_metrics
                    and hasattr(st.session_state, "bg_metrics")
                    and st.session_state.bg_metrics):
                m = st.session_state.bg_metrics
                scene_type = m.get("scene_type", "N/A")
                scene_color = "#10B981" if scene_type == "SIMPLE" else "#F59E0B"
                st.markdown(f"""
                <div style="margin-top:1.2rem;">
                    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
                        <span style="font-size:0.75rem; font-weight:700; color:#6B7280;
                              text-transform:uppercase; letter-spacing:1px;">Pipeline Metrics</span>
                        <span style="background:{scene_color}22; border:1px solid {scene_color}44;
                              color:{scene_color}; padding:0.2rem 0.7rem; border-radius:100px;
                              font-size:0.7rem; font-weight:700;">{scene_type}</span>
                    </div>
                    <div class="metric-row">
                        <div class="metric-box">
                            <div class="value">{m.get('stage2_birefnet_ms', 0):.0f}ms</div>
                            <div class="label">BiRefNet SOD</div>
                        </div>
                        <div class="metric-box">
                            <div class="value">{m.get('stage3_routing_ms', 0):.1f}ms</div>
                            <div class="label">Scene Routing</div>
                        </div>
                        <div class="metric-box">
                            <div class="value">{m.get('stage3b_sam_ms', 0):.0f}ms</div>
                            <div class="label">SAM Segment</div>
                        </div>
                        <div class="metric-box">
                            <div class="value">{m.get('stage4_matting_ms', 0):.0f}ms</div>
                            <div class="label">MODNet Alpha</div>
                        </div>
                        <div class="metric-box">
                            <div class="value">{m.get('stage5_postprocess_ms', 0):.0f}ms</div>
                            <div class="label">Postprocess</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            # ── Just show the uploaded image ──
            st.markdown('<div class="img-label">📷 Uploaded Image</div>', unsafe_allow_html=True)
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(img_rgb, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
            oh, ow = img_rgb.shape[:2]
            st.caption(f"{ow} × {oh} px • Ready to process")

    else:
        # ── Empty state ──
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; color:#4B5563;">
            <div style="font-size:3rem; margin-bottom:1rem;">📁</div>
            <div style="font-size:1.1rem; font-weight:600; color:#6B7280;">
                Upload an image to get started
            </div>
            <div style="font-size:0.85rem; margin-top:0.5rem;">
                Supports JPG, PNG, BMP, WEBP — up to 200MB
            </div>
        </div>
        """, unsafe_allow_html=True)

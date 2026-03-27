import streamlit as st
import os
import sys
import torch
import cv2
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.enhance import HybridEnhancer
from pipeline.preprocess import preprocess_image
from pipeline.postprocess import postprocess_tensor
from utils.image_utils import bytes_to_image

# Page config
st.set_page_config(page_title="AI Hybrid Color Correction", layout="wide")
st.title("🌟 AI Hybrid Color Correction System")
st.caption("Powered by Pretrained Zero-DCE++ (Exposure) & Restormer (Details) | Optimized for Mid-Range GPUs")

# Initialize models
@st.cache_resource
def load_enhancer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HybridEnhancer(device=device)

enhancer = load_enhancer()

# Sidebar options
st.sidebar.header("Settings")
use_restormer = st.sidebar.checkbox("Enable Restormer (Detail Refinement)", value=True, help="Refines details and removes noise, but requires more VRAM and compute time.")

# Main UI
uploaded_file = st.file_uploader("Upload an Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to numpy array
    img_rgb = bytes_to_image(uploaded_file.read())
    
    st.write(f"**Original Size:** {img_rgb.shape[1]}x{img_rgb.shape[0]}")
    
    if st.button("Enhance Image 🚀", use_container_width=True):
        with st.spinner("Processing... Please wait."):
            # 1. Preprocess
            tensor = preprocess_image(img_rgb, max_size=512)
            
            # 2. Enhance
            out_tensor = enhancer.enhance(tensor, use_restormer=use_restormer)
            
            # 3. Postprocess
            out_rgb = postprocess_tensor(out_tensor)
            
        # Display side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Original Image", use_column_width=True)
        with col2:
            st.image(out_rgb, caption="Enhanced Image", use_column_width=True)
            
        # Download button
        st.success("Enhancement Complete!")
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
        if is_success:
            st.download_button(
                label="📥 Download Enhanced Image",
                data=buffer.tobytes(),
                file_name="enhanced_image.png",
                mime="image/png",
                use_container_width=True
            )

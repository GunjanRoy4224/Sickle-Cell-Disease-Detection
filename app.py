import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.model import get_model
from src.data_setup import get_transforms
from src.explainability import reshape_transform
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sickle Cell Disease Detection",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHE MODEL LOADING ---
@st.cache_resource
def load_diagnostic_model(model_name):
    # Map friendly names to timm architectures
    arch_name = 'swin_base_patch4_window7_224' if 'swin' in model_name.lower() else 'resnet50'
    model = get_model(arch_name, pretrained=False, num_classes=1)
    
    weights_path = f"best_model_{arch_name}.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
        return model, arch_name
    else:
        st.error(f"⚠️ Weights file not found: {weights_path}. Please train the model first.")
        return None, arch_name

# --- HELPER FUNCTION FOR GRAD CAM ---
def get_cam_image(model, arch_name, input_tensor, display_img):
    if 'swin' in arch_name.lower():
        target_layer = model.layers[-2].blocks[-1].norm2
        cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    else:
        target_layer = model.layer4[-1]
        cam = GradCAM(model=model, target_layers=[target_layer])
    
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(
        input_tensor=input_tensor, 
        targets=targets, 
        aug_smooth=True, 
        eigen_smooth=True
    )[0]
    
    rgb_img = np.array(display_img).astype(np.float32) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization

# --- UI LAYOUT ---
st.title("🩸 Sickle Cell Anemia Disease Detection")
st.markdown("""
This research demo classifies blood-smear images with deep learning. **It is not a medical device and must not be used for diagnosis.**
Upload a microscopic blood smear image to receive a prediction alongside a **Grad-CAM visual audit**, to help inspect whether the model attends to cells rather than background artifacts.
""")

# Sidebar
st.sidebar.header("🔧 Configuration")
selected_model = st.sidebar.selectbox(
    "Select Architecture",
    ("Swin Transformer (Base)", "ResNet50 (Baseline)")
)

# Default thresholds: written by main.py (chosen on the validation split). Legacy values are only a fallback.
import json
from pathlib import Path
_thr_file = Path("results/thresholds.json")
_thr = json.loads(_thr_file.read_text()) if _thr_file.exists() else {}
if 'Swin' in selected_model:
    default_threshold = _thr.get("swin_base_patch4_window7_224", 0.8950)
else:
    default_threshold = _thr.get("resnet50", 0.2913)

decision_threshold = st.sidebar.slider(
    "Decision threshold (targets 95% recall on validation data)",
    min_value=0.0,
    max_value=1.0,
    value=default_threshold,
    step=0.01,
    help="Adjusting this probability threshold trades off False Positives vs False Negatives. The default is chosen on the validation split to reach 95% recall; it is not a guarantee on new data."
)

st.sidebar.markdown("---")
st.sidebar.info("Upload an image in the main panel to begin inference.")

# Main inference panel
uploaded_file = st.file_uploader("Upload Blood Smear Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display original uploaded image
    raw_img = Image.open(uploaded_file).convert('RGB')
    
    # 2. Load Model
    with st.spinner(f"Loading {selected_model}..."):
        model, arch_name = load_diagnostic_model(selected_model)
    
    if model is not None:
        # 3. Preprocess Image
        _, val_tfms = get_transforms()
        
        # Center crop exactly as done in explainability.py
        width, height = raw_img.size
        crop_size = min(width, height)
        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = (width + crop_size) / 2
        bottom = (height + crop_size) / 2
        cropped_img = raw_img.crop((left, top, right, bottom))
        
        # We need the 224x224 display image for Grad-CAM overlay
        display_img = cropped_img.resize((224, 224))
        
        input_tensor = val_tfms(display_img).unsqueeze(0) # Add batch dimension

        # 4. Inference
        with st.spinner("Analyzing image..."):
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
        
        # 5. Grad CAM Visualization
        with st.spinner("Generating Explainability Heatmap..."):
            cam_visualization = get_cam_image(model, arch_name, input_tensor, display_img)
        
        # 6. Display Results
        st.markdown("### 📊 Diagnostic Results")
        
        prediction = "Sickle Cell Detected" if prob >= decision_threshold else "Healthy / No Sickle Cell"
        color = "red" if prob >= decision_threshold else "green"
        
        st.markdown(f"**Prediction:** <span style='color:{color}; font-size:24px;'>**{prediction}**</span>", unsafe_allow_html=True)
        st.progress(prob, text=f"Confidence Score: {prob:.4f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(raw_img, caption="Original Uploaded Image", use_container_width=True)
        with col2:
            st.image(display_img, caption="Preprocessed (Cropped & Resized)", use_container_width=True)
        with col3:
            st.image(cam_visualization, caption="Grad-CAM Auditing", use_container_width=True)
            
        if prob >= decision_threshold:
            st.warning("⚠️ **Clinical Alert:** Geometric anomalies matching Sickle Cell profiles detected. See Grad-CAM heatmaps for regions of interest.", icon="⚠️")
        else:
            st.success("✅ **Clear:** No sickle cell anomalies detected exceeding the clinical threshold.", icon="✅")


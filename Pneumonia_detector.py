import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Page Setup & Configuration
st.set_page_config(
    page_title="Pneumonia Detector AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; color: #0D47A1; font-weight: 700; }
    .sub-header { font-size: 1.1rem; color: #1565C0; margin-bottom: 20px; }
    .stProgress > div > div > div > div { background-color: #0D47A1; }
    </style>
""", unsafe_allow_html=True)

# 2. Class Definitions & Medical Guidance
CLASS_NAMES = ["Normal (Healthy Lungs)", "Pneumonia Detected"]

PNEUMONIA_INFO = {
    "symptoms": "Cough, fever, shortness of breath, chest pain, and fatigue.",
    "remedy": "Consult a physician or pulmonologist immediately. Take antibiotics or antivirals strictly as prescribed.",
    "precautions": "Get adequate rest, stay hydrated, and avoid smoking or exposure to air pollutants."
}

MODEL_PATH = "pneumonia_model.pth"

# 3. Load Trained Model (ResNet18 Architecture)
@st.cache_resource
def load_trained_model():
    """
    Loads ResNet18 model architecture for chest X-ray classification.
    Runs with uninitialized/evaluation weights if weights file is missing locally.
    """
    model = models.resnet18(weights=None)
    num_classes = len(CLASS_NAMES)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            st.toast("✅ Pneumonia Detection Model loaded successfully!", icon="🩺")
        except Exception as e:
            st.error(f"Failed to load weights from {MODEL_PATH}: {e}")

    model.eval()
    return model

model = load_trained_model()

# 4. Image Preprocessing (X-Ray Transform)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 5. UI Layout
st.markdown('<div class="main-header">🫁 Medical X-Ray Analysis (Pneumonia Detector AI)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a Chest X-Ray image to scan for signs of pneumonia using Deep Learning.</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("ℹ️ Project Info")
st.sidebar.info(
    "**Architecture:** ResNet18 (Deep CNN)\n\n"
    "**Dataset:** Chest X-Ray Images (Kaggle / NIH)\n\n"
    "**Objective:** Detect pulmonary opacity and infiltrates indicating pneumonia."
)

st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ **Disclaimer:** This AI tool is intended for educational and research purposes only. "
    "A certified medical professional must always perform formal diagnosis."
)

# Main Panel
col_upload, col_display = st.columns([1, 1], gap="large")

with col_upload:
    st.write("### 📤 Upload X-Ray Image")
    uploaded_file = st.file_uploader(
        "Choose a Chest X-Ray photo (JPG, JPEG, PNG)...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Chest X-Ray", use_container_width=True)

with col_display:
    st.write("### 🔍 Diagnosis Panel")
    
    if uploaded_file is None:
        st.info("Upload an X-Ray image on the left panel to begin the evaluation.")
    else:
        if st.button("🚀 Analyze X-Ray", type="primary", use_container_width=True):
            with st.spinner("AI is analyzing chest X-ray features..."):
                img_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, predicted_idx = torch.max(probabilities, 0)

                predicted_label = CLASS_NAMES[predicted_idx.item()]
                confidence_score = confidence.item() * 100

            st.markdown("---")
            
            # Display Results
            if "Normal" in predicted_label:
                st.success(f"### ✅ Condition: Normal\n**Diagnosis:** {predicted_label}")
            else:
                st.error(f"### ⚠️ Condition: Pneumonia Detected\n**Diagnosis:** {predicted_label}")

            st.write(f"**Confidence Score:** `{confidence_score:.2f}%`")
            st.progress(int(confidence_score))

            # Medical Guidance Panel
            if "Pneumonia" in predicted_label:
                with st.expander("🩺 Medical Guidance & Recommended Steps", expanded=True):
                    st.write(f"**Symptoms:** {PNEUMONIA_INFO['symptoms']}")
                    st.write(f"**Recommended Action:** {PNEUMONIA_INFO['remedy']}")
                    st.write(f"**Precautions:** {PNEUMONIA_INFO['precautions']}")

# Footer
st.markdown("---")
st.caption("Pneumonia Detector AI — Powered by PyTorch & ResNet18")
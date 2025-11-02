import streamlit as st
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image
import json
import pandas as pd
import os
import requests

# ======================================================
# CONFIGURATION
# ======================================================
MODEL_REPO = "MalaikaNaveed1/food-recognition-vit"  # Hugging Face repo
LOCAL_CACHE = "./hf_cache"  # Folder to cache downloaded files locally

# Additional visual/metrics files (optional)
RESULTS_FILE = "results_summary.json"
REPORT_FILE = "classification_report.csv"
CONFUSION_FILE = "confusion_matrix.png"
SAMPLES_FILE = "sample_predictions.png"
DISTRIBUTION_FILE = "dataset_distribution.png"

# Ensure cache directory exists
os.makedirs(LOCAL_CACHE, exist_ok=True)

# ======================================================
# LOAD MODEL AND PROCESSOR FROM HUGGING FACE
# ======================================================
@st.cache_resource
def load_model_from_hub():
    try:
        # Load model and processor from your HF repo
        model = ViTForImageClassification.from_pretrained(
            MODEL_REPO,
            use_safetensors=True
        )
        processor = ViTImageProcessor.from_pretrained(MODEL_REPO)
        
        id2label = model.config.id2label
        return model, processor, id2label
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {str(e)}")
        return None, None, None

model, processor, id2label = load_model_from_hub()

if model is None:
    st.stop()

# ======================================================
# APP HEADER
# ======================================================
st.title("🍽️ Food Classification using ViT-Base (Hugging Face)")
st.markdown("### Fine-tuned `google/vit-base-patch16-224` on Food-101 dataset")
st.caption("Model: [MalaikaNaveed1/food-recognition-vit](https://huggingface.co/MalaikaNaveed1/food-recognition-vit)")

# Sidebar navigation
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", ["Overview", "Visualizations", "Predict"])

# ======================================================
# OVERVIEW SECTION
# ======================================================
if section == "Overview":
    st.subheader("📋 Model Overview")

    # Try loading results summary
    results_path = os.path.join(LOCAL_CACHE, RESULTS_FILE)
    if not os.path.exists(results_path):
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{RESULTS_FILE}"
        r = requests.get(url)
        if r.status_code == 200:
            with open(results_path, "wb") as f:
                f.write(r.content)

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        st.json(results)
    else:
        st.warning("Results summary not found on Hugging Face repo.")

    st.markdown("#### Classification Report (Top 10 classes)")
    report_path = os.path.join(LOCAL_CACHE, REPORT_FILE)
    if not os.path.exists(report_path):
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{REPORT_FILE}"
        r = requests.get(url)
        if r.status_code == 200:
            with open(report_path, "wb") as f:
                f.write(r.content)

    if os.path.exists(report_path):
        df = pd.read_csv(report_path)
        df_filtered = df[~df.iloc[:, 0].isin(['accuracy', 'macro avg', 'weighted avg'])]
        st.dataframe(df_filtered.sort_values("f1-score", ascending=False).head(10))
    else:
        st.warning("Classification report not found.")

# ======================================================
# VISUALIZATIONS SECTION
# ======================================================
elif section == "Visualizations":
    st.subheader("📊 Dataset & Model Insights")

    def load_image_from_hub(filename):
        path = os.path.join(LOCAL_CACHE, filename)
        if not os.path.exists(path):
            url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{filename}"
            r = requests.get(url)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
        return path if os.path.exists(path) else None

    col1, col2 = st.columns(2)
    with col1:
        img_path = load_image_from_hub(DISTRIBUTION_FILE)
        if img_path:
            st.image(img_path, caption="Class Distribution", use_container_width=True)
        else:
            st.warning("Dataset distribution image not found.")

    with col2:
        img_path = load_image_from_hub(CONFUSION_FILE)
        if img_path:
            st.image(img_path, caption="Confusion Matrix", use_container_width=True)
        else:
            st.warning("Confusion matrix image not found.")

    img_path = load_image_from_hub(SAMPLES_FILE)
    if img_path:
        st.image(img_path, caption="Sample Predictions", use_container_width=True)
    else:
        st.warning("Sample predictions image not found.")

# ======================================================
# PREDICTION SECTION
# ======================================================
elif section == "Predict":
    st.subheader("🔍 Try the Model Yourself!")

    uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Analyzing image..."):
                inputs = processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    top_probs, top_labels = probs.topk(5)

            st.markdown("### 🍽️ Top Predictions:")
            for i, (p, l) in enumerate(zip(top_probs[0], top_labels[0]), 1):
                label = id2label[l.item()]
                confidence = p.item() * 100
                st.write(f"**{i}. {label.replace('_', ' ').title()}**")
                st.progress(confidence / 100)
                st.write(f"Confidence: {confidence:.2f}%")
                st.write("")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("Fine-tuned ViT-base for Food Classification | Powered by Hugging Face Transformers 🚀")

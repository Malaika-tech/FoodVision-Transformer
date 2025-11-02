import streamlit as st
from transformers import ViTForImageClassification, ViTFeatureExtractor
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
LOCAL_CACHE = "./hf_cache"  # Cache folder for visuals

RESULTS_FILE = "results_summary.json"
REPORT_FILE = "classification_report.csv"
CONFUSION_FILE = "confusion_matrix.png"
SAMPLES_FILE = "sample_predictions.png"
DISTRIBUTION_FILE = "dataset_distribution.png"

os.makedirs(LOCAL_CACHE, exist_ok=True)

# ======================================================
# LOAD LABELS FROM LOCAL CONFIG.JSON
# ======================================================
def load_labels_from_config():
    """Load id2label mapping from local config.json file."""
    config_path = "config.json"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            if "id2label" in config:
                # Convert string keys to integers
                id2label = {int(k): v for k, v in config["id2label"].items()}
                st.sidebar.success(f"✓ Loaded {len(id2label)} labels from config.json")
                return id2label
            else:
                st.sidebar.error("⚠️ 'id2label' not found in config.json")
                return None
        except Exception as e:
            st.sidebar.error(f"Error reading config.json: {e}")
            return None
    else:
        st.sidebar.error("⚠️ config.json file not found in directory")
        return None

# ======================================================
# LOAD MODEL AND FEATURE EXTRACTOR
# ======================================================
@st.cache_resource
def load_model():
    try:
        model = ViTForImageClassification.from_pretrained(
            MODEL_REPO,
            num_labels=101,
            ignore_mismatched_sizes=True
        )
        feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_REPO)
        
        # Load labels from local config.json
        id2label = load_labels_from_config()
        
        # Fallback if config.json doesn't have labels
        if id2label is None or len(id2label) == 0:
            st.sidebar.warning("Using model's default labels")
            id2label = model.config.id2label if hasattr(model.config, 'id2label') else {}
        
        if len(id2label) == 0:
            st.sidebar.error("No labels found! Please check your config.json")
            id2label = {i: f"Class_{i}" for i in range(101)}
        
        return model, feature_extractor, id2label
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, feature_extractor, id2label = load_model()
if model is None:
    st.stop()

# Display label info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Classes:** {len(id2label)}")
if st.sidebar.checkbox("Show all labels"):
    st.sidebar.write(id2label)

# ======================================================
# APP HEADER
# ======================================================
st.title("🍽️ Food Classification using ViT-Base")
st.markdown("### Fine-tuned `vit-base-patch16-224` on Food-101 dataset")
st.caption(f"Model: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})")

# Sidebar navigation
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", ["Overview", "Visualizations", "Predict"])

# ======================================================
# OVERVIEW SECTION
# ======================================================
if section == "Overview":
    st.subheader("📋 Model Overview")

    # Load results summary
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
        st.warning("Results summary not found.")

    # Classification report
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
                inputs = feature_extractor(images=image, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    top_probs, top_labels = probs.topk(5)

            st.markdown("### 🍽️ Top Predictions:")
            for i, (p, l) in enumerate(zip(top_probs[0], top_labels[0]), 1):
                label_idx = l.item()
                # Get the food name from config.json mapping
                label_name = id2label.get(label_idx, f"Unknown_Class_{label_idx}")
                confidence = p.item() * 100
                
                # Display with progress bar
                st.markdown(f"**{i}. {label_name}**")
                st.progress(confidence / 100)
                st.caption(f"Confidence: {confidence:.2f}%")
                st.markdown("")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.exception(e)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("Fine-tuned ViT-base for Food Classification | Powered by Hugging Face Transformers 🚀")
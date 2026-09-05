import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

# Configure Streamlit page
st.set_page_config(
    page_title="RetinaGuard - AI Diabetic Retinopathy Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and modern UI
st.markdown("""
<style>
    /* Dark glassmorphic theme styling */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Header card */
    .main-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .main-header h1 {
        color: #38bdf8;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 8px;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    /* Grade Result Badges */
    .badge-no-dr {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .badge-mild {
        background-color: rgba(234, 179, 8, 0.2);
        color: #facc15;
        border: 1px solid #eab308;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }

    .badge-moderate {
        background-color: rgba(249, 115, 22, 0.2);
        color: #fb923c;
        border: 1px solid #f97316;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }

    .badge-severe, .badge-proliferative {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }

    /* Clinical warning box */
    .disclaimer-box {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 14px;
        margin-top: 20px;
        color: #fca5a5;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

APP_ROOT = Path(__file__).resolve().parent
MODEL_PATH = APP_ROOT / 'model' / 'best_model.keras'
IMAGE_SIZE = 224
CROP_THRESHOLD = 10
CROP_PADDING = 10

CLASS_NAMES = [
    'No DR',
    'Mild',
    'Moderate',
    'Severe',
    'Proliferative'
]

# Keras 3 compatibility patch for quantization_config
import keras
_orig_dense_init = keras.layers.Dense.__init__
keras.layers.Dense.__init__ = lambda self, *args, quantization_config=None, **kwargs: _orig_dense_init(self, *args, **kwargs)

@st.cache_resource
def load_dr_model():
    """Load and cache the TensorFlow / Keras DR model."""
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}")
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_dr_model()

def preprocess_image_file(image_path):
    """Preprocess image for model input."""
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)

    gray = tf.reduce_mean(image, axis=-1)
    mask = gray > tf.cast(CROP_THRESHOLD / 255.0, tf.float32)
    coords = tf.where(mask)

    def crop_image():
        y_min = tf.cast(tf.reduce_min(coords[:, 0]), tf.int32)
        y_max = tf.cast(tf.reduce_max(coords[:, 0]), tf.int32)
        x_min = tf.cast(tf.reduce_min(coords[:, 1]), tf.int32)
        x_max = tf.cast(tf.reduce_max(coords[:, 1]), tf.int32)

        h = tf.cast(tf.shape(image)[0], tf.int32)
        w = tf.cast(tf.shape(image)[1], tf.int32)

        y_min = tf.maximum(y_min - CROP_PADDING, 0)
        x_min = tf.maximum(x_min - CROP_PADDING, 0)
        y_max = tf.minimum(y_max + CROP_PADDING + 1, h)
        x_max = tf.minimum(x_max + CROP_PADDING + 1, w)

        return image[y_min:y_max, x_min:x_max]

    image = tf.cond(tf.size(coords) > 0, crop_image, lambda: image)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method='bilinear')
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def assess_quality(image_path):
    """Assess fundus image quality."""
    image = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    arr = image.numpy()
    h, w = arr.shape[:2]
    gray = np.mean(arr, axis=2)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    black_fraction = float(np.mean(gray < 0.04))

    non_black = gray >= 0.04
    if np.any(non_black):
        ys, xs = np.where(non_black)
        fov_fraction = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) / max(h * w, 1)
    else:
        fov_fraction = 0.0

    flags = []
    if h < 100 or w < 100: flags.append('very_small_image')
    if brightness < 0.05: flags.append('very_dark')
    if brightness > 0.95: flags.append('very_bright')
    if contrast < 0.05: flags.append('very_low_contrast')

    score = 1.0
    penalties = {'very_small_image': 0.40, 'very_dark': 0.15, 'very_bright': 0.15, 'very_low_contrast': 0.20}
    for flag in flags:
        score -= penalties.get(flag, 0.0)
    score = float(np.clip(score, 0.0, 1.0))

    status = 'Acceptable' if score >= 0.70 else ('Review Needed' if score >= 0.40 else 'Ungradable')
    return {'status': status, 'score': score, 'brightness': brightness, 'contrast': contrast, 'flags': flags}

def predict_dr(image_path):
    """Run model prediction on preprocessed image."""
    img_tensor = preprocess_image_file(image_path)
    batch = tf.expand_dims(img_tensor, axis=0)
    probs = model(batch, training=False).numpy()[0]
    probs = np.asarray(probs, dtype=np.float64)
    pred_idx = int(np.argmax(probs))
    return pred_idx, CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs

# Top Header
st.markdown("""
<div class="main-header">
    <h1>👁️ RetinaGuard</h1>
    <p>AI-Powered Explainable Diabetic Retinopathy Screening Platform</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Model Overview")
    st.markdown("""
    - **Architecture**: `EfficientNetB0`
    - **Input Resolution**: `224 x 224`
    - **Classifier**: 5-Class Severity Scale
    - **Model File**: `best_model.keras`
    """)
    st.divider()
    st.subheader("Severity Scale")
    st.markdown("""
    0. **No DR**: Normal retina
    1. **Mild**: Microaneurysms only
    2. **Moderate**: Hemorrhages & hard exudates
    3. **Severe**: Cotton wool spots & venous beading
    4. **Proliferative**: Neovascularization
    """)

# Main Upload Section
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Retinal Image Upload")
    uploaded_file = st.file_uploader(
        "Upload a Retinal Fundus Photograph (.jpg, .png, .jpeg)",
        type=["jpg", "png", "jpeg"],
        help="Upload high quality retinal fundus photograph for automated screening"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image ({image.width} x {image.height})", use_column_width=True)

with col2:
    st.subheader("📊 Diagnostic Results")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing Retinal Image & Assessing Quality..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = Path(tmp.name)

            try:
                quality = assess_quality(tmp_path)
                pred_idx, grade, confidence, probs = predict_dr(tmp_path)

                # Quality Metric Cards
                q_col1, q_col2, q_col3 = st.columns(3)
                with q_col1:
                    st.metric("Quality Status", quality['status'])
                with q_col2:
                    st.metric("Quality Score", f"{quality['score']*100:.0f}%")
                with q_col3:
                    st.metric("Contrast", f"{quality['contrast']:.2f}")

                st.divider()

                # Classification Result Badge
                badge_class = f"badge-{grade.lower().replace(' ', '-')}"
                st.markdown(f'<div class="{badge_class}">Screening Result: {grade}</div>', unsafe_allow_html=True)
                st.write("")
                st.progress(confidence, text=f"Confidence Level: {confidence * 100:.1f}%")

                st.subheader("Probability Distribution Across 5 Grades")
                chart_data = pd.DataFrame({
                    "Grade": CLASS_NAMES,
                    "Probability (%)": [round(p * 100, 2) for p in probs]
                })
                st.bar_chart(chart_data.set_index("Grade"), color="#38bdf8")

            finally:
                tmp_path.unlink(missing_ok=True)
    else:
        st.info("👆 Please upload a retinal fundus image on the left panel to begin automated screening.")

# Clinical Disclaimer Footer
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Clinical Disclaimer:</strong><br>
    This AI tool is designed as an advisory screening support system and does NOT constitute a formal medical diagnosis. 
    All screening outputs must be validated by a qualified ophthalmologist or trained healthcare professional.
</div>
""", unsafe_allow_html=True)

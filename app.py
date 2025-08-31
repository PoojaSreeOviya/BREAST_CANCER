import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model once
@st.cache_resource
def load_trained_model():
    return load_model("vgg16_model.keras")

model = load_trained_model()

# Class mapping
classes = {0: "Benign", 1: "Malignant", 2: "Normal"}

def predict_img(img):
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)
    return classes[pred_class], confidence

# ===================== UI =====================
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .main {background-color: #f7f9fc;}
    .stButton>button {
        background-color: #2E86C1;
        color:white;
        font-size:16px;
        border-radius:8px;
        padding:10px 20px;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Breast Cancer Detection using VGG16")
st.write("Upload an ultrasound image to predict whether it is **Benign, Malignant, or Normal**.")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        pred, conf = predict_img(img)
        st.markdown(
            f"""
            <div class="prediction-card">
                <h2>🔎 Prediction: <span style="color:#2E86C1">{pred}</span></h2>
                <h3>Confidence: {conf*100:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Extra insights
    st.info("⚠️ Note: This tool is for **educational purposes only** and should not replace professional medical diagnosis.")

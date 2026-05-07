import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from streamlit_lottie import st_lottie

# ===================== CONFIG =====================
PATH = r"C:\Users\Admin\Desktop\Improved Brain Tumor Segmentation in MR Images with modified unet\unet_webapp\checkpoints\checkpoints\unet_p6_best.h5"

st.set_page_config(page_title="UNet Segmentation", page_icon="🧠", layout="centered")

# Load Lottie animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json")

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model(path):
    model = tf.keras.models.load_model(path, compile=False)
    return model

model = load_model(PATH)
input_height, input_width = model.input_shape[1], model.input_shape[2]
input_channels = model.input_shape[3]

# ===================== CUSTOM CSS =====================
st.markdown("""
    <style>
    body { background: linear-gradient(135deg, #1e1e2f, #2b2b40); color: #fff; font-family: 'Poppins', sans-serif;}
    .stApp { background-color: #f0f4f8; }
    h1, h2, h3 { text-align: center; color: #00e0ff; text-shadow: 0px 0px 8px rgba(0, 224, 255, 0.5);}
    .css-1v3fvcr, .css-1kyxreq, .stButton>button {
        border-radius: 12px !important;
        font-size: 16px !important;
        background: linear-gradient(90deg, #00e0ff, #0072ff);
        color: White !important;
        border: none;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover { transform: scale(1.05); box-shadow: 0px 0px 15px rgba(0, 224, 255, 0.6);}
    .stFileUploader label { color: #00e0ff !important; font-weight: 600; font-size: 18px;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown("<h1>🧠 UNet Brain Tumor Segmentation App</h1>", unsafe_allow_html=True)
st.markdown("<h3>Upload an image and visualize segmentation results instantly!</h3>", unsafe_allow_html=True)
st_lottie(lottie_ai, height=200, key="ai")

# ===================== UPLOAD & PREDICT =====================
uploaded = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded)

    # Convert to model-required channels
    if input_channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    # Resize to model input
    img = img.resize((input_width, input_height))

    st.image(img, caption="📸 Uploaded Image", use_column_width=True)

    # Preprocess
    x = np.array(img) / 255.0
    if input_channels == 1:
        x = np.expand_dims(x, axis=-1)
    x = np.expand_dims(x, axis=0)

    # Predict
    with st.spinner("⏳ Running segmentation..."):
        pred = model.predict(x)[0]
        pred_mask = np.squeeze(pred)
        st.success("✅ Prediction complete!")

        # Display uploaded and mask side by side
        st.markdown("### Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.image(pred_mask, caption="Predicted Mask", use_column_width=True)

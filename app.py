
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

st.title("MR Image Segmentation - UNet P6")

MODEL_PATH = "./checkpoints/unet_p6_best.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()
IMG_SIZE = (224, 224)

uploaded_file = st.file_uploader("Upload MR image (.png/.jpg)", type=["png", "jpg", "jpeg"])
threshold = st.slider("Segmentation Threshold", 0.0, 1.0, 0.5, 0.01)

def preprocess_image(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_norm = img_resized.astype('float32') / 255.0
    return img_rgb, img_norm

def predict_mask(model, img_norm, threshold=0.5):
    pred = model.predict(np.expand_dims(img_norm, 0))[0,...,0]
    pred_bin = (pred > threshold).astype(np.uint8)
    return pred, pred_bin

def display_results(img_rgb, pred, pred_bin):
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(img_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    ax[1].imshow(img_rgb)
    ax[1].imshow(pred, cmap='jet', alpha=0.5)
    ax[1].contour(pred_bin, colors='white', linewidths=0.5)
    ax[1].set_title("Segmentation Overlay")
    ax[1].axis('off')
    st.pyplot(fig)

if uploaded_file:
    img_rgb, img_norm = preprocess_image(uploaded_file)
    pred, pred_bin = predict_mask(model, img_norm, threshold)
    display_results(img_rgb, pred, pred_bin)

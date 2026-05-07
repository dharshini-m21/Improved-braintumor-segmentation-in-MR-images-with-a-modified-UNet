# 🧠 Brain Tumor Detection using Improved U-Net

## 📌 Overview
This project focuses on detecting brain tumors from MRI images using an improved U-Net deep learning model. It aims to provide faster and more accurate tumor identification compared to manual analysis.

## 🚀 Features
- Improved U-Net architecture
- Leaky ReLU activation function
- High accuracy (99.4%)
- Dice Score: 90.2%
- Fast training (~19 minutes using GPU)

## 🛠️ Technologies Used
- Python
- TensorFlow / PyTorch
- OpenCV
- NumPy
- Matplotlib

## 📂 Dataset
MRI brain scan images used for training, validation, and testing.

## ⚙️ Workflow
1. Input MRI image
2. Image preprocessing
3. Tumor segmentation using U-Net
4. Output with highlighted tumor region

## ▶️ Installation & Usage
```bash
git clone <your-repo-link>
cd project-folder
pip install -r requirements.txt
python train.py
python predict.py

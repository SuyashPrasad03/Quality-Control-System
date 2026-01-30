# 🏭 VisionGuard: Industrial Defect Detection System

![TensorFlow](https://img.shields.io/badge/TensorFlow-Lite-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Status-Production-green)

**VisionGuard** is an automated optical inspection (AOI) system designed for manufacturing lines. It utilizes Deep Learning (Transfer Learning with MobileNetV2) to detect defects in casting products in real-time. The model is optimized using **TensorFlow Lite** for low-latency inference on edge devices (like Raspberry Pi or factory tablets).

## 🚀 Key Features
- **High Accuracy:** Achieves >92% accuracy in distinguishing between defective and non-defective casting parts.
- **Edge Optimized:** Model quantized to **TensorFlow Lite (TFLite)** format, reducing size by 4x for deployment on low-power hardware.
- **Real-Time Dashboard:** Interactive web interface built with **Streamlit** for instant visual feedback.
- **Data Efficient:** Uses Transfer Learning to perform effectively even with limited training data.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow, Keras, MobileNetV2
- **Optimization:** TensorFlow Lite (Quantization)
- **Frontend:** Streamlit
- **Image Processing:** OpenCV, PIL

## 📂 Project Structure
```bash
quality_control_system/
├── dataset/             # Training images (Ignored in Git)
├── model/
│   ├── quality_model.tflite  # Optimized Edge Model
├── src/
│   ├── train.py         # Transfer Learning training script
│   ├── convert.py       # TFLite conversion & quantization script
│   └── app.py           # Streamlit Dashboard
├── requirements.txt
└── README.md
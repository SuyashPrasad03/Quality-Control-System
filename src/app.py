import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- PATH SETUP ---
# Robust way to find the model relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(project_root, 'model', 'quality_model.tflite')

st.set_page_config(page_title="VisionGuard AI", page_icon="🏭")
st.title("🏭 VisionGuard: Quality Control System")

# Debug Info (Visible only if something goes wrong)
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ CRITICAL ERROR: Model file not found!")
    st.code(f"Looking for model at: {MODEL_PATH}")
    st.write("Files in current directory:")
    st.write(os.listdir(current_dir))
    st.stop()

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

interpreter = load_model()

if interpreter is None:
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- IMAGE PROCESSING ---
def process_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- UI ---
st.write("Upload an image of a casting part for inspection.")
uploaded_file = st.file_uploader("Choose image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Part', width=300)
    
    if st.button('Analyze Quality'):
        with st.spinner('Analyzing...'):
            input_data = process_image(image)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            prediction_score = output_data[0][0]
            
            st.divider()
            
            # 0 = Defective, 1 = OK (Based on folder alphabet order: def_front, ok_front)
            if prediction_score < 0.5:
                confidence = (1 - prediction_score) * 100
                st.error(f"🚨 **DEFECT DETECTED**")
                st.metric("Confidence", f"{confidence:.2f}%")
            else:
                confidence = prediction_score * 100
                st.success(f"✅ **PART IS OK**")
                st.metric("Confidence", f"{confidence:.2f}%")
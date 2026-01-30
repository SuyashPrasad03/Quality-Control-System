import tensorflow as tf
import os

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
model_h5_path = os.path.join(project_root, 'model', 'quality_model.h5')
model_tflite_path = os.path.join(project_root, 'model', 'quality_model.tflite')

print(f"Loading model from: {model_h5_path}")

# --- CONVERSION ---
try:
    model = tf.keras.models.load_model(model_h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(model_tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"SUCCESS: TFLite model saved to {model_tflite_path}")

except Exception as e:
    print(f"ERROR: Could not convert model. {e}")
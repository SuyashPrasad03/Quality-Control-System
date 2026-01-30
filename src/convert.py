import tensorflow as tf
import os

# Paths
model_path = os.path.join(os.path.dirname(__file__), '../model/quality_model.h5')
tflite_path = os.path.join(os.path.dirname(__file__), '../model/quality_model.tflite')

print("Loading Keras model...")
model = tf.keras.models.load_model(model_path)

# Convert to TensorFlow Lite
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# OPTIMIZATION: Quantization
# This flag tells TF to optimize the model size (make it 4x smaller)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the TFLite model
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"Success! TFLite model saved to {tflite_path}")
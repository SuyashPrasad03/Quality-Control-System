import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- PATH SETUP ---
# Get the absolute path of THIS script (src/train.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up to the project root
project_root = os.path.dirname(current_dir)
# Define absolute paths
DATA_DIR = os.path.join(project_root, 'dataset')
MODEL_DIR = os.path.join(project_root, 'model')

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print(f"Training Data located at: {DATA_DIR}")
print(f"Model will be saved to: {MODEL_DIR}")

# --- CONFIG ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- DATA GENERATORS ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# --- MODEL BUILDING ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- TRAINING ---
print("Starting training...")
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)

# --- SAVING ---
save_path = os.path.join(MODEL_DIR, 'quality_model.h5')
model.save(save_path)
print(f"SUCCESS: Model saved to {save_path}")
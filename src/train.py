import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. Setup Configuration
IMG_SIZE = (224, 224)  # Standard size for MobileNet
BATCH_SIZE = 32
DATA_DIR = os.path.join(os.path.dirname(__file__), '../dataset')

# 2. Prepare Data (Augmentation)
# We artificially create "more" data by flipping and rotating images
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values (0-1)
    validation_split=0.2,   # Use 20% of data for checking accuracy
    horizontal_flip=True,
    rotation_range=20
)

print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',    # Binary because we only have OK vs DEF
    subset='training'
)

print("Loading validation data...")
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 3. Build the Model (Transfer Learning)
# Load MobileNetV2 but cut off the "head" (include_top=False)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model (we don't want to ruin what it already knows)
base_model.trainable = False

# Add our own "head" for casting defects
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)  # Prevent overfitting
predictions = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Train
print("Starting training...")
history = model.fit(
    train_generator,
    epochs=5,  # 5 epochs is usually enough for Transfer Learning
    validation_data=validation_generator
)

# 6. Save the Model (CORRECTED PATH)
# We use os.path.join to ensure it saves exactly where we want, no matter where you run the script from
save_dir = os.path.join(os.path.dirname(__file__), '../model')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, 'quality_model.h5')
model.save(model_path)
print(f"Model successfully saved to: {model_path}")
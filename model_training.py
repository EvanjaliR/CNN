# Import required libraries
import tensorflow as tf
import numpy as np

# Print TensorFlow version (must be 2.18)
print("TensorFlow Version:", tf.__version__)

"""
------------------------------------------------------------
PART 1 - DATA PREPROCESSING
------------------------------------------------------------
Step 1: Organize your dataset in the following folder structure:

dataset\
├── train\
│   ├── good\
│   └── defective\
└── test\
    ├── good\
    └── defective\


Each image should be placed in the corresponding class folder.
"""

# Load and preprocess training data
train_set = tf.keras.utils.image_dataset_from_directory(
    'dataset/train',
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary'
)

# Data augmentation to improve model generalization
train_data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),        # Normalize pixel values
    tf.keras.layers.RandomFlip('horizontal'), # Random horizontal flip
    tf.keras.layers.RandomZoom(0.2),          # Zoom-in effects
    tf.keras.layers.RandomRotation(0.2),      # Random rotation
    tf.keras.layers.RandomShear(0.2)          # Shear transformation
])

# Apply augmentation to the training dataset
augmented_train_dataset = train_set.map(lambda x, y: (train_data_augmentation(x, training=True), y))

# Load and preprocess test data (no augmentation, just normalization)
test_set = tf.keras.utils.image_dataset_from_directory(
    'dataset/test',
    image_size=(64, 64),
    batch_size=32,
    label_mode='binary',
    shuffle=False
)

test_data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255)
])

augmented_test_dataset = test_set.map(lambda x, y: (test_data_augmentation(x, training=False), y))

"""
------------------------------------------------------------
PART 2 - BUILDING THE CNN MODEL
------------------------------------------------------------
This is a basic Convolutional Neural Network architecture.
You can add more layers or tune it later for better accuracy.
"""

cnn = tf.keras.models.Sequential([
    # First convolutional layer
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPool2D(pool_size=2),

    # Second convolutional layer
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2),

    # Flattening the output of conv layers
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(units=128, activation='relu'),

    # Output layer for binary classification
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""
------------------------------------------------------------
PART 3 - TRAINING THE CNN
------------------------------------------------------------
The model will train for 25 epochs and show accuracy and loss.
"""

cnn.fit(
    x=augmented_train_dataset,
    validation_data=augmented_test_dataset,
    epochs=25
)

# Save the trained model to disk
model_save_path = 'your_trained_model'
cnn.save(model_save_path)
print(f"✅ Model saved to: {model_save_path}")
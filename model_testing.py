import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('your_trained_model.keras')


# Load and preprocess the sample image
img_path = 'single_prediction/sample1.jpg'
img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
print("Prediction Score:", prediction[0][0])

# Interpret result
if prediction[0][0] > 0.5:
    print("Predicted Class: Defective Tyre")
else:
    print("Predicted Class: Good Tyre")

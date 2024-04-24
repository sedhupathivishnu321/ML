import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.applications import MobileNetV2

# Load pre-trained MobileNetV2 model (you can use other pre-trained models too)
model = MobileNetV2(weights='imagenet')

# Define a function to preprocess images for the model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Define a function to predict whether the leaf is infected or not
def predict_infection(img_path):
    preprocessed_img = preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    # Check if the top prediction indicates an infected leaf
    for _, label, confidence in decoded_predictions:
        if label == 'infected_leaf' and confidence > 0.5:  # Adjust threshold as needed
            return True
    return False

# Example usage
if predict_infection('C:\Users\sedhu\OneDrive\Pictures\KALMAN.png'):
    print("The leaf is infected.")
else:
    print("The leaf is not infected.")

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained chessboard corner detection model
corner_detection_model = load_model('chessboard_corner_detection_model.h5')

def detect_chessboard_corners(image):
    # Preprocess the image
    processed_image = cv2.resize(image, (224, 224))  # Resize to match the model's input size
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

    # Perform corner detection using the trained model
    corners = corner_detection_model.predict(processed_image)

    # Postprocess the detected corners
    corners = np.squeeze(corners)  # Remove the batch dimension
    corners = corners.reshape((-1, 2))  # Reshape to (num_corners, 2)

    return corners
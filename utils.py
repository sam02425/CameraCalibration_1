import cv2
import pickle

def load_calibration(file_path):
    with open(file_path, 'rb') as file:
        calibration_data = pickle.load(file)
    return calibration_data

def draw_bounding_boxes(image, detected_objects):
    for x1, y1, x2, y2, label in detected_objects:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

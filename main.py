import cv2
from config import *
from utils import load_calibration, draw_bounding_boxes
from object_detection import ObjectDetector, transform_coordinates

def main():
    # Load camera calibration parameters
    left_camera_matrix, left_dist_coeffs = load_calibration(LEFT_CALIBRATION_FILE)
    right_camera_matrix, right_dist_coeffs = load_calibration(RIGHT_CALIBRATION_FILE)

    # Load stereo calibration parameters
    R, T = load_calibration(STEREO_CALIBRATION_FILE)

    # Initialize object detector
    detector = ObjectDetector(YOLO_CONFIG, YOLO_WEIGHTS, COCO_NAMES)

    # Initialize left and right cameras
    left_camera = cv2.VideoCapture(LEFT_CAMERA_ID)
    right_camera = cv2.VideoCapture(RIGHT_CAMERA_ID)

    while True:
        # Capture frames from left and right cameras
        ret_left, left_frame = left_camera.read()
        ret_right, right_frame = right_camera.read()

        if not ret_left or not ret_right:
            break

        # Detect objects in left and right frames
        left_detected_objects = detector.detect(left_frame)
        right_detected_objects = detector.detect(right_frame)

        # Extract object coordinates from detection results
        left_detected_points = np.array([[x1, y1], [x2, y2]] for x1, y1, x2, y2, _ in left_detected_objects)
        right_detected_points = np.array([[x1, y1], [x2, y2]] for x1, y1, x2, y2, _ in right_detected_objects)

        # Transform object coordinates to world coordinate system
        left_object_points_world = transform_coordinates(left_detected_points, left_camera_matrix, left_dist_coeffs, R, T, np.array([-12, 0, -6]))
        right_object_points_world = transform_coordinates(right_detected_points, right_camera_matrix, right_dist_coeffs, R, T, np.array([12, 0, -6]))

        # Combine transformed object coordinates
        object_points_world = np.concatenate((left_object_points_world, right_object_points_world), axis=0)

        # Perform further processing or visualization with object

import cv2
import numpy as np
import pickle

class ObjectDetector:
    def __init__(self, yolo_config, yolo_weights, coco_names):
        self.net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
        self.classes = []
        with open(coco_names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.output_layers = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, image):
        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detected_objects = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                detected_objects.append((x, y, x + w, y + h, label))

        return detected_objects

def transform_coordinates(detected_points, camera_matrix, dist_coeffs, rotation_matrix, translation_vector, offset):
    # Undistort detected points
    undistorted_points = cv2.undistortPoints(detected_points, camera_matrix, dist_coeffs)

    # Convert undistorted points to homogeneous coordinates
    undistorted_points = np.squeeze(undistorted_points, axis=1).astype(np.float32)
    undistorted_points_homogeneous = np.hstack((undistorted_points, np.ones((undistorted_points.shape[0], 1), dtype=np.float32)))

    # Convert rotation matrix and translation vector to 3x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = (translation_vector.flatten() + offset)[:3]
    transformation_matrix = transformation_matrix[:3, :]  # Extract the first 3 rows

    # Transform object coordinates from camera coordinate system to world coordinate system
    object_points_world_homogeneous = []
    for point in undistorted_points_homogeneous:
        transformed_point = np.dot(point, transformation_matrix)
        object_points_world_homogeneous.append(transformed_point)

    object_points_world_homogeneous = np.array(object_points_world_homogeneous)
    object_points_world = object_points_world_homogeneous[:, :3]

    return object_points_world

def main():
    # Load camera calibration parameters
    with open('left_calibration.pkl', 'rb') as file:
        left_camera_matrix, left_dist_coeffs = pickle.load(file)
    with open('right_calibration.pkl', 'rb') as file:
        right_camera_matrix, right_dist_coeffs = pickle.load(file)

    # Load stereo calibration parameters
    with open('stereo_calibration.pkl', 'rb') as file:
        R, T = pickle.load(file)

    # Initialize object detector
    yolo_config = 'yolo.cfg'
    yolo_weights = 'yolo.weights'
    coco_names = 'coco.names'
    detector = ObjectDetector(yolo_config, yolo_weights, coco_names)

    # Initialize left and right cameras
    left_camera = cv2.VideoCapture(0)
    right_camera = cv2.VideoCapture(1)

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

        # Perform further processing or visualization with object_points_world

        # Display the frames
        cv2.imshow('Left Frame', left_frame)
        cv2.imshow('Right Frame', right_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the cameras and close windows
    left_camera.release()
    right_camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

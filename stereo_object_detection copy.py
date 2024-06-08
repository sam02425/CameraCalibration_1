import cv2
import numpy as np
import pickle
from collections import deque

# Load YOLOv4 network
net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load stereo calibration parameters
with open('left_calibration.pkl', 'rb') as file:
    left_camera_matrix, left_dist_coeffs = pickle.load(file)
with open('right_calibration.pkl', 'rb') as file:
    right_camera_matrix, right_dist_coeffs = pickle.load(file)

left_rotation_matrix = np.load('left_rotation_matrix.npy')
left_translation_vector = np.load('left_translation_vector.npy')
right_rotation_matrix = np.load('right_rotation_matrix.npy')
right_translation_vector = np.load('right_translation_vector.npy')

# Initialize object tracking variables
tracked_objects = {}
next_object_id = 1
max_lost_frames = 10
detected_objects = {}  # Dictionary to store detected objects and their count

# Define the region of interest (ROI) for the platform
roi_x1, roi_y1 = 100, 100  # Top-left corner coordinates
roi_x2, roi_y2 = 500, 400  # Bottom-right corner coordinates

def detect_objects(image):
    # Perform object detection using YOLOv4
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the detection results
    class_ids = []
    confidences = []
    boxes = []
    detections = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust the confidence threshold as needed
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                width = int(detection[2] * image.shape[1])
                height = int(detection[3] * image.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                detections.append((center_x, center_y))

    # Apply non-maximum suppression to remove overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Filter the detections, class IDs, and boxes based on the NMS indices
    detections = [detections[i] for i in indices]
    class_ids = [class_ids[i] for i in indices]
    boxes = [boxes[i] for i in indices]

    return detections, class_ids, boxes

def visualize_objects(left_image, right_image, points_3d, left_class_ids, right_class_ids, left_boxes, right_boxes, classes):
    # Stitch the left and right images
    stitched_image = np.hstack((left_image, right_image))

    # Draw the ROI rectangle
    roi_color = (0, 0, 255)  # Red color for the ROI
    roi_thickness = 2
    cv2.rectangle(stitched_image, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, roi_thickness)

    # Visualize objects on the stitched image
    for point, class_id, box in zip(points_3d, left_class_ids + right_class_ids, left_boxes + right_boxes):
        x, y, z = point
        label = classes[class_id]
        color = (0, 255, 0)  # Green color for bounding box and text
        cv2.rectangle(stitched_image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
        cv2.putText(stitched_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(stitched_image, (int(x), int(y)), 5, color, -1)
        cv2.putText(stitched_image, f"({x:.2f}, {y:.2f}, {z:.2f})", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow('Detected Objects', stitched_image)

def track_objects(left_detections, right_detections, left_class_ids, right_class_ids, classes):
    global next_object_id

    all_detections = left_detections + right_detections
    all_class_ids = left_class_ids + right_class_ids

    # Update tracked objects based on current detections
    for detection, class_id in zip(all_detections, all_class_ids):
        x, y = detection
        # Check if the detection is within the ROI
        if roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2:
            found = False
            for object_id, tracked_object in tracked_objects.items():
                if np.linalg.norm(np.array(detection) - np.array(tracked_object['last_detection'])) < 50:
                    tracked_object['trace'].append(detection)
                    tracked_object['last_detection'] = detection
                    tracked_object['lost_frames'] = 0
                    found = True
                    break
            if not found:
                tracked_objects[next_object_id] = {
                    'class_id': class_id,
                    'trace': deque(maxlen=20),
                    'last_detection': detection,
                    'lost_frames': 0
                }
                tracked_objects[next_object_id]['trace'].append(detection)
                next_object_id += 1

    # Remove tracked objects that have been lost for too many frames or are outside the ROI
    lost_objects = []
    for object_id, tracked_object in tracked_objects.items():
        x, y = tracked_object['last_detection']
        tracked_object['lost_frames'] += 1
        if tracked_object['lost_frames'] > max_lost_frames or not (roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2):
            lost_objects.append(object_id)
            class_id = tracked_object['class_id']
            class_name = classes[class_id]
            if class_name in detected_objects:
                detected_objects[class_name] -= 1
    for object_id in lost_objects:
        del tracked_objects[object_id]

def main():
    # Initialize left and right cameras
    left_camera = cv2.VideoCapture(1)  # Adjust the camera index if necessary
    right_camera = cv2.VideoCapture(2)  # Adjust the camera index if necessary

    # Set camera resolution (optional)
    left_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    left_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    right_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    right_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Check if cameras are opened successfully
    if not left_camera.isOpened() or not right_camera.isOpened():
        print("Failed to open cameras.")
        return

    # Get the class labels
    with open('classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    while True:
        # Capture synchronized images from cameras
        _, left_image = left_camera.read()
        _, right_image = right_camera.read()

        # Perform object detection on left and right images
        left_detections, left_class_ids, left_boxes = detect_objects(left_image)
        right_detections, right_class_ids, right_boxes = detect_objects(right_image)

        # Before object detection
        cv2.imshow('Right Camera', right_image)

        # After object detection
        for box in right_boxes:
            cv2.rectangle(right_image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        cv2.imshow('Right Camera (Detected)', right_image)

        # Check if left_detections and right_detections are empty
        if len(left_detections) == 0 and len(right_detections) == 0:
            print("No object detections found in both images.")
            cv2.imshow('Detected Objects', np.hstack((left_image, right_image)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Convert detections to float32 and reshape to (n, 1, 2)
        left_points = np.array(left_detections, dtype=np.float32).reshape(-1, 1, 2)
        right_points = np.array(right_detections, dtype=np.float32).reshape(-1, 1, 2)

        # Ensure left_points and right_points have the correct shape
        if len(left_points) == 0:
            left_points = np.empty((0, 1, 2), dtype=np.float32)
        if len(right_points) == 0:
            right_points = np.empty((0, 1, 2), dtype=np.float32)

        # Undistort and rectify the detected object points
        left_points_undistorted = cv2.undistortPoints(left_points, left_camera_matrix, left_dist_coeffs, R=left_rotation_matrix, P=left_camera_matrix)
        right_points_undistorted = cv2.undistortPoints(right_points, right_camera_matrix, right_dist_coeffs, R=right_rotation_matrix, P=right_camera_matrix)

        # Ensure an equal number of points in left and right images
        min_points = min(len(left_points_undistorted), len(right_points_undistorted))
        left_points_undistorted = left_points_undistorted[:min_points]
        right_points_undistorted = right_points_undistorted[:min_points]

        if min_points > 0:
            # Create projection matrices for left and right cameras
            left_projection_matrix = np.hstack((left_rotation_matrix, left_translation_vector))
            right_projection_matrix = np.hstack((right_rotation_matrix, right_translation_vector))

            # Triangulate the object points
            points_4d = cv2.triangulatePoints(left_projection_matrix, right_projection_matrix, left_points_undistorted, right_points_undistorted)

            if points_4d.size > 0:
                # Convert homogeneous coordinates to 3D points
                points_3d_homogeneous = cv2.convertPointsFromHomogeneous(points_4d.T).squeeze()

                # Ensure points_3d_homogeneous has shape (n, 3)
                if points_3d_homogeneous.ndim == 1:
                    points_3d_homogeneous = points_3d_homogeneous.reshape(1, -1)

                # Transform the triangulated object points to world coordinates
                points_3d_world = np.dot(left_rotation_matrix, points_3d_homogeneous.T).T + left_translation_vector.T

                # Track objects within the ROI
                track_objects(left_detections, right_detections, left_class_ids, right_class_ids, classes)

                # Visualize the detected objects on the stitched image
                visualize_objects(left_image, right_image, points_3d_world, left_class_ids, right_class_ids, left_boxes, right_boxes, classes)

                # Clear the terminal
                import os
                os.system('cls' if os.name == 'nt' else 'clear')

                # Show class name and count on the terminal
                print("Detected Objects:")
                for class_name, count in detected_objects.items():
                    print(f"{class_name}: {count}")
                print(f"Total detected objects: {sum(detected_objects.values())}")
            else:
                print("Triangulation failed. No points found.")
        else:
            print("No matching object detections found in both images.")

        # Check for 'q' key press to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the cameras
    left_camera.release()
    right_camera.release()

    # Close all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
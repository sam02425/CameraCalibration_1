import cv2
import numpy as np
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from calibration_app import CalibrationApp
from PyQt5.QtWidgets import QApplication
import sys

# Define the platform's coordinate system
platform_origin = np.array([0, 0, 0])  # X, Y, Z coordinates of the origin
platform_x_axis = np.array([1, 0, 0])  # X-axis direction
platform_y_axis = np.array([0, 1, 0])  # Y-axis direction
platform_z_axis = np.array([0, 0, 1])  # Z-axis direction

# Load intrinsic and extrinsic parameters
with open('left_calibration.pkl', 'rb') as file:
    left_camera_matrix, left_dist_coeffs = pickle.load(file)
with open('right_calibration.pkl', 'rb') as file:
    right_camera_matrix, right_dist_coeffs = pickle.load(file)

left_rotation_matrix = np.load('left_rotation_matrix.npy')
left_translation_vector = np.load('left_translation_vector.npy')
right_rotation_matrix = np.load('right_rotation_matrix.npy')
right_translation_vector = np.load('right_translation_vector.npy')

# Simulate object detection for each camera (replace with your actual detection code)
left_detected_points = np.array([[100, 200], [300, 400], [500, 600], [700, 800], [900, 1000]], dtype=np.float32)
right_detected_points = np.array([[150, 250], [350, 450], [550, 650], [750, 850], [950, 1050]], dtype=np.float32)

def transform_coordinates_to_platform(detected_points, camera_matrix, dist_coeffs, rotation_matrix, translation_vector):
    # Undistort the detected object coordinates
    undistorted_points = cv2.undistortPoints(detected_points, camera_matrix, dist_coeffs)

    # Squeeze the undistorted points to remove the extra dimension
    undistorted_points = np.squeeze(undistorted_points)

    # Convert the undistorted points to homogeneous coordinates
    undistorted_points_homogeneous = np.hstack((undistorted_points, np.ones((undistorted_points.shape[0], 1))))
    undistorted_points_homogeneous = np.hstack((undistorted_points_homogeneous, np.ones((undistorted_points_homogeneous.shape[0], 1))))

    # Construct the transformation matrix
    transformation_matrix = np.hstack((rotation_matrix, translation_vector))
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))

    # Transform the object coordinates to the platform's coordinate system
    object_points_platform_homogeneous = np.dot(undistorted_points_homogeneous, transformation_matrix.T)

    # Convert homogeneous coordinates back to Cartesian coordinates
    object_points_platform = object_points_platform_homogeneous[:, :3]

    return object_points_platform

# Transform object coordinates to the platform's coordinate system for each camera
left_object_points_platform = transform_coordinates_to_platform(left_detected_points, left_camera_matrix, left_dist_coeffs, left_rotation_matrix, left_translation_vector)
right_object_points_platform = transform_coordinates_to_platform(right_detected_points, right_camera_matrix, right_dist_coeffs, right_rotation_matrix, right_translation_vector)

# Combine the transformed object coordinates from all cameras
all_object_points_platform = np.concatenate((left_object_points_platform, right_object_points_platform), axis=0)

# Remove duplicate or overlapping points from different camera views
distance_threshold = 1000.0  # Increase the threshold value further
unique_mask = np.ones(len(all_object_points_platform), dtype=bool)
for i in range(len(all_object_points_platform)):
    if unique_mask[i]:
        distances = np.linalg.norm(all_object_points_platform - all_object_points_platform[i], axis=1)
        unique_mask[distances < distance_threshold] = False
unique_object_points_platform = all_object_points_platform[unique_mask]

# Apply clustering algorithm (e.g., K-means) if there are enough samples
num_clusters = 3
if len(unique_object_points_platform) >= num_clusters:
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(unique_object_points_platform)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
else:
    print("Skipping clustering step due to insufficient samples.")
    labels = None
    cluster_centers = None

# Visualize the 3D object positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(unique_object_points_platform[:, 0], unique_object_points_platform[:, 1], unique_object_points_platform[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Object Positions in Platform Coordinate System')
plt.show()

# Load the estimated calibration parameters and fine-tuned model
calibration_params = np.load('calibration_params.npy')
unsupervised_calibration_params = np.load('unsupervised_calibration_params.npy')
fine_tuned_model = load_model('fine_tuned_calibration_model.h5')

# Use the loaded calibration parameters and fine-tuned model for further processing
# ...

# Launch the user-friendly calibration app
app = QApplication(sys.argv)
calibration_app = CalibrationApp()
calibration_app.show()
sys.exit(app.exec_())
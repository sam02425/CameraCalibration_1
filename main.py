import cv2
import numpy as np
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the platform's coordinate system
platform_origin = np.array([0, 0, 0])  # X, Y, Z coordinates of the origin
platform_x_axis = np.array([1, 0, 0])  # X-axis direction
platform_y_axis = np.array([0, 1, 0])  # Y-axis direction
platform_z_axis = np.array([0, 0, 1])  # Z-axis direction

# Load camera calibration parameters for each camera
with open('/home/luuser1/Desktop/CameraCalibration_1/left_calibration.pkl', 'rb') as file:
    cam1_camera_matrix, cam1_dist_coeffs = pickle.load(file)
with open('/home/luuser1/Desktop/CameraCalibration_1/right_calibration.pkl', 'rb') as file:
    cam2_camera_matrix, cam2_dist_coeffs = pickle.load(file)

cam1_rotation_matrix = np.load('/home/luuser1/Desktop/CameraCalibration_1/cam1_rotation_matrix.npy')
cam1_translation_vector = np.load('/home/luuser1/Desktop/CameraCalibration_1/cam1_translation_vector.npy')
cam2_rotation_matrix = np.load('/home/luuser1/Desktop/CameraCalibration_1/cam2_rotation_matrix.npy')
cam2_translation_vector = np.load('/home/luuser1/Desktop/CameraCalibration_1/cam2_translation_vector.npy')

# Ensure the translation vectors have the correct shape
cam1_translation_vector = cam1_translation_vector.reshape(3, 1)
cam2_translation_vector = cam2_translation_vector[:, 2].reshape(3, 1)

# Simulate object detection for each camera (replace with your actual detection code)
cam1_detected_points = np.array([[100, 200], [300, 400], [500, 600], [700, 800], [900, 1000]], dtype=np.float32)
cam2_detected_points = np.array([[150, 250], [350, 450], [550, 650], [750, 850], [950, 1050]], dtype=np.float32)

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

print("Cam1 camera matrix:")
print(cam1_camera_matrix)
print("Cam1 distortion coefficients:")
print(cam1_dist_coeffs)
print("Cam1 rotation matrix:")
print(cam1_rotation_matrix)
print("Cam1 translation vector:")
print(cam1_translation_vector)

print("Cam2 camera matrix:")
print(cam2_camera_matrix)
print("Cam2 distortion coefficients:")
print(cam2_dist_coeffs)
print("Cam2 rotation matrix:")
print(cam2_rotation_matrix)
print("Cam2 translation vector:")
print(cam2_translation_vector)

# Transform object coordinates to the platform's coordinate system for each camera
cam1_object_points_platform = transform_coordinates_to_platform(cam1_detected_points, cam1_camera_matrix, cam1_dist_coeffs, cam1_rotation_matrix, cam1_translation_vector)
cam2_object_points_platform = transform_coordinates_to_platform(cam2_detected_points, cam2_camera_matrix, cam2_dist_coeffs, cam2_rotation_matrix, cam2_translation_vector)

print("Cam1 object points in platform coordinates:")
print(cam1_object_points_platform)
print("Cam2 object points in platform coordinates:")
print(cam2_object_points_platform)

# Combine the transformed object coordinates from all cameras
all_object_points_platform = np.concatenate((cam1_object_points_platform, cam2_object_points_platform), axis=0)

print("Number of samples before removing duplicates:", len(all_object_points_platform))

# Remove duplicate or overlapping points from different camera views
distance_threshold = 1000.0  # Increase the threshold value further
unique_mask = np.ones(len(all_object_points_platform), dtype=bool)
for i in range(len(all_object_points_platform)):
    if unique_mask[i]:
        distances = np.linalg.norm(all_object_points_platform - all_object_points_platform[i], axis=1)
        unique_mask[distances < distance_threshold] = False
unique_object_points_platform = all_object_points_platform[unique_mask]


print("Number of samples after removing duplicates:", len(unique_object_points_platform))

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

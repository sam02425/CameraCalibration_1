import cv2
import numpy as np
import glob
import pickle

################ CAMERA CALIBRATION #############################

def calibrate_camera(images, chessboard_size, square_size):
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return camera_matrix, dist_coeffs

################ COORDINATE TRANSFORMATION #############################

def transform_coordinates(detected_points, camera_matrix, dist_coeffs, rotation_matrix, translation_vector, offset):
    # Undistort detected points
    undistorted_points = cv2.undistortPoints(detected_points, camera_matrix, dist_coeffs)

    # Transform object coordinates to camera coordinate system
    object_points_cam = cv2.convertPointsToHomogeneous(undistorted_points)

    # Transform object coordinates to world coordinate system
    object_points_world = cv2.transformPointsForward(rotation_matrix, translation_vector + offset, object_points_cam)

    return object_points_world

################ MAIN FUNCTION #############################

def main():
    # Chessboard dimensions
    chessboard_size = (9, 6)  # Number of inner corners: (columns, rows)
    square_size = 0.025  # Size of each square in meters

    # Calibrate left camera
    left_images = glob.glob('left_images/*.png')
    left_camera_matrix, left_dist_coeffs = calibrate_camera(left_images, chessboard_size, square_size)

    # Calibrate right camera
    right_images = glob.glob('right_images/*.png')
    right_camera_matrix, right_dist_coeffs = calibrate_camera(right_images, chessboard_size, square_size)

    # Camera positions relative to the center point (in inches)
    left_offset = np.array([-12, 0, -6])  # 12 inches left, 6 inches back
    right_offset = np.array([12, 0, -6])  # 12 inches right, 6 inches back

    # Rotation and translation matrices (replace with actual values)
    left_rotation_matrix = np.eye(3)
    left_translation_vector = np.zeros((3, 1))
    right_rotation_matrix = np.eye(3)
    right_translation_vector = np.zeros((3, 1))

    # Object detection (replace with actual object detection code)
    left_detected_points = np.array([[100, 200], [300, 400], [500, 600]], dtype=np.float32)
    right_detected_points = np.array([[150, 250], [350, 450], [550, 650]], dtype=np.float32)

    # Transform object coordinates for left camera
    left_object_points_world = transform_coordinates(left_detected_points, left_camera_matrix, left_dist_coeffs,
                                                     left_rotation_matrix, left_translation_vector, left_offset)

    # Transform object coordinates for right camera
    right_object_points_world = transform_coordinates(right_detected_points, right_camera_matrix, right_dist_coeffs,
                                                      right_rotation_matrix, right_translation_vector, right_offset)

    # Combine transformed object coordinates
    object_points_world = np.concatenate((left_object_points_world, right_object_points_world), axis=0)

    # Print the transformed object coordinates
    print("Transformed object coordinates in world coordinate system:")
    print(object_points_world)

if __name__ == '__main__':
    main()
# extrinsic_para.py: Calculate extrinsic parameters for stereo camera calibration

import cv2
import numpy as np
import glob
import pickle
import os

# Debug: Print current working directory
print("Current working directory:", os.getcwd())

# Load intrinsic parameters
left_calibration_file = 'left_calibration.pkl'
right_calibration_file = 'right_calibration.pkl'

print("Loading left calibration file:", left_calibration_file)
with open(left_calibration_file, 'rb') as file:
    cam1_camera_matrix, cam1_dist_coeffs = pickle.load(file)

print("Loading right calibration file:", right_calibration_file)
with open(right_calibration_file, 'rb') as file:
    cam2_camera_matrix, cam2_dist_coeffs = pickle.load(file)

# Chessboard dimensions
chessboard_size = (9, 6)  # Number of inner corners: (columns, rows)
frame_size = (640, 480)  # Camera frame size

# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints_left = []  # 2D points in left camera image plane
imgpoints_right = []  # 2D points in right camera image plane

# Get the list of calibration images
left_images = sorted(glob.glob('left_images/*.png'))
right_images = sorted(glob.glob('right_images/*.png'))

# Take the minimum number of images from both folders
num_images = min(len(left_images), len(right_images))

# Create a directory to save the images with detected corners
output_dir = 'detected_corners'
os.makedirs(output_dir, exist_ok=True)

# Define the termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

image_index = 0
for left_image, right_image in zip(left_images[:num_images], right_images[:num_images]):
    img_left = cv2.imread(left_image)
    img_right = cv2.imread(right_image)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        objpoints.append(objp)
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

        # Save the images with detected corners
        left_output_image = os.path.join(output_dir, f'left_{image_index}.png')
        right_output_image = os.path.join(output_dir, f'right_{image_index}.png')
        cv2.drawChessboardCorners(img_left, chessboard_size, corners_left, ret_left)
        cv2.drawChessboardCorners(img_right, chessboard_size, corners_right, ret_right)
        cv2.imwrite(left_output_image, img_left)
        cv2.imwrite(right_output_image, img_right)

        image_index += 1

print(f"Number of images used for calibration: {len(objpoints)}")
print(f"Number of image points (left camera): {len(imgpoints_left)}")
print(f"Number of image points (right camera): {len(imgpoints_right)}")

# Perform stereo camera calibration
flags = cv2.CALIB_FIX_INTRINSIC

if len(objpoints) > 0 and len(imgpoints_left) > 0 and len(imgpoints_right) > 0:
    try:
        retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right, cam1_camera_matrix, cam1_dist_coeffs,
            cam2_camera_matrix, cam2_dist_coeffs, frame_size, criteria=criteria, flags=flags)
        if not retval:
            raise ValueError("Stereo calibration failed.")
    except cv2.error as e:
        print("Stereo calibration error:", str(e))
        raise e

    # Save extrinsic parameters
    np.save('left_rotation_matrix.npy', R)
    np.save('left_translation_vector.npy', T)
    np.save('right_rotation_matrix.npy', R)
    np.save('right_translation_vector.npy', T)

    print("Stereo camera calibration completed successfully.")
else:
    print("Not enough valid image points for stereo calibration.")

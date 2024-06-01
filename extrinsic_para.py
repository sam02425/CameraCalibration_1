import cv2
import numpy as np
import glob
import pickle
import os

# Load intrinsic parameters
with open('/home/luuser1/Desktop/CameraCalibration_1/left_calibration.pkl', 'rb') as file:
    cam1_camera_matrix, cam1_dist_coeffs = pickle.load(file)
with open('/home/luuser1/Desktop/CameraCalibration_1/right_calibration.pkl', 'rb') as file:
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
left_images = sorted(glob.glob('/home/luuser1/Desktop/CameraCalibration_1/left_images/*.png'))
right_images = sorted(glob.glob('/home/luuser1/Desktop/CameraCalibration_1/right_images/*.png'))

# Take the minimum number of images from both folders
num_images = min(len(left_images), len(right_images))

# Create a directory to save the images with detected corners
output_dir = '/home/luuser1/Desktop/CameraCalibration_1/detected_corners'
os.makedirs(output_dir, exist_ok=True)

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
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

if len(objpoints) > 0 and len(imgpoints_left) > 0 and len(imgpoints_right) > 0:
    retval, _, _, _, _, cam1_rotation_matrix, cam1_translation_vector, cam2_rotation_matrix, cam2_translation_vector = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, cam1_camera_matrix, cam1_dist_coeffs, cam2_camera_matrix, cam2_dist_coeffs, frame_size, criteria=criteria, flags=flags)

    # Save extrinsic parameters
    np.save('/home/luuser1/Desktop/CameraCalibration_1/cam1_rotation_matrix.npy', cam1_rotation_matrix)
    np.save('/home/luuser1/Desktop/CameraCalibration_1/cam1_translation_vector.npy', cam1_translation_vector)
    np.save('/home/luuser1/Desktop/CameraCalibration_1/cam2_rotation_matrix.npy', cam2_rotation_matrix)
    np.save('/home/luuser1/Desktop/CameraCalibration_1/cam2_translation_vector.npy', cam2_translation_vector)

    print("Stereo camera calibration completed successfully.")
else:
    print("Not enough valid image points for stereo calibration.")

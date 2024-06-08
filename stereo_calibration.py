# stereo_calibration.py is used to calibrate the stereo camera system.

import cv2
import numpy as np
import pickle
import glob

def main():
    # Chessboard dimensions
    chessboard_size = (9, 6)  # Number of inner corners: (columns, rows)
    frame_size = (640, 480)  # Camera frame size

    # Load camera calibration results
    with open('left_calibration.pkl', 'rb') as file:
        left_camera_matrix, left_dist_coeffs = pickle.load(file)
    with open('right_calibration.pkl', 'rb') as file:
        right_camera_matrix, right_dist_coeffs = pickle.load(file)

    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    left_imgpoints = []  # 2D points in left image plane
    right_imgpoints = []  # 2D points in right image plane

    # Get the list of stereo calibration images
    left_images = glob.glob('left_images/*.png')
    right_images = glob.glob('right_images/*.png')

    for left_img, right_img in zip(left_images, right_images):
        left = cv2.imread(left_img)
        right = cv2.imread(right_img)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

        if ret_left and ret_right:
            objpoints.append(objp)
            left_imgpoints.append(corners_left)
            right_imgpoints.append(corners_right)

    # Perform stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    try:
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints,
                                                        left_camera_matrix, left_dist_coeffs,
                                                        right_camera_matrix, right_dist_coeffs,
                                                        frame_size, criteria=criteria, flags=flags)
        if not ret:
            raise ValueError("Stereo calibration failed.")
    except cv2.error as e:
        print("Stereo calibration error:", str(e))
        raise e

    # Save the stereo calibration results
    with open('stereo_calibration.pkl', 'wb') as file:
        pickle.dump((R, T), file)

    # Obtain rotation and translation matrices from stereo calibration
    left_rotation_matrix_actual = R
    left_translation_vector_actual = T[:, :3]

    # For the right camera
    right_rotation_matrix_actual = R
    right_translation_vector_actual = T[:, 3:]

    # Save rotation and translation matrices
    np.save('left_rotation_matrix.npy', left_rotation_matrix_actual)
    np.save('left_translation_vector.npy', left_translation_vector_actual)
    np.save('right_rotation_matrix.npy', right_rotation_matrix_actual)
    np.save('right_translation_vector.npy', right_translation_vector_actual)

if __name__ == '__main__':
    main()


import cv2
import numpy as np
import glob
import pickle

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
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints,
                                                      left_camera_matrix, left_dist_coeffs,
                                                      right_camera_matrix, right_dist_coeffs,
                                                      frame_size, criteria=criteria, flags=flags)

    # Save the stereo calibration results
    with open('stereo_calibration.pkl', 'wb') as file:
        pickle.dump((R, T), file)

if __name__ == '__main__':
    main()

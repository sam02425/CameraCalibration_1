import cv2
import numpy as np
import glob
import pickle

def calibrate_camera(images_folder, chessboard_size, frame_size):
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Get the list of calibration images
    images = glob.glob(f'{images_folder}/*.png')

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

    return camera_matrix, dist_coeffs

def main():
    # Chessboard dimensions
    chessboard_size = (9, 6)  # Number of inner corners: (columns, rows)
    frame_size = (640, 480)  # Camera frame size

    # Calibrate left camera
    left_camera_matrix, left_dist_coeffs = calibrate_camera('left_images', chessboard_size, frame_size)
    print("Left Camera Calibration Done!")

    # Calibrate right camera
    right_camera_matrix, right_dist_coeffs = calibrate_camera('right_images', chessboard_size, frame_size)
    print("Right Camera Calibration Done!")

    # Save the camera calibration results
    with open('left_calibration.pkl', 'wb') as file:
        pickle.dump((left_camera_matrix, left_dist_coeffs), file)
    with open('right_calibration.pkl', 'wb') as file:
        pickle.dump((right_camera_matrix, right_dist_coeffs), file)

if __name__ == '__main__':
    main()

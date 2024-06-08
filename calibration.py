
import numpy as np
import cv2 as cv
import glob
import pickle

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9, 6)
frameSize = (640, 480)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 26
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
left_objpoints = []  # 3d point in real world space for left camera
left_imgpoints = []  # 2d points in image plane for left camera.
right_objpoints = []  # 3d point in real world space for right camera
right_imgpoints = []  # 2d points in image plane for right camera.

left_images = glob.glob('left_images/*.png')
right_images = glob.glob('right_images/*.png')

for left_image, right_image in zip(left_images, right_images):
    # Process left camera image
    left_img = cv.imread(left_image)
    left_gray = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners for left camera
    ret_left, corners_left = cv.findChessboardCorners(left_gray, chessboardSize, None)

    # If found, add object points, image points (after refining them) for left camera
    if ret_left:
        left_objpoints.append(objp)
        corners_left_refined = cv.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
        left_imgpoints.append(corners_left_refined)

    # Process right camera image
    right_img = cv.imread(right_image)
    right_gray = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners for right camera
    ret_right, corners_right = cv.findChessboardCorners(right_gray, chessboardSize, None)

    # If found, add object points, image points (after refining them) for right camera
    if ret_right:
        right_objpoints.append(objp)
        corners_right_refined = cv.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)
        right_imgpoints.append(corners_right_refined)

    # Draw and display the corners for both left and right cameras (optional)
    if ret_left and ret_right:
        cv.drawChessboardCorners(left_img, chessboardSize, corners_left_refined, ret_left)
        cv.drawChessboardCorners(right_img, chessboardSize, corners_right_refined, ret_right)
        cv.imshow('Left Image', left_img)
        cv.imshow('Right Image', right_img)
        cv.waitKey(1000)

cv.destroyAllWindows()

############## CALIBRATION #######################################################

# Calibrate left camera
ret_left, cameraMatrix_left, dist_left, rvecs_left, tvecs_left = cv.calibrateCamera(
    left_objpoints, left_imgpoints, frameSize, None, None)

# Save the left camera calibration result for later use
pickle.dump((cameraMatrix_left, dist_left), open("left_calibration.pkl", "wb"))

# Calibrate right camera
ret_right, cameraMatrix_right, dist_right, rvecs_right, tvecs_right = cv.calibrateCamera(
    right_objpoints, right_imgpoints, frameSize, None, None)

# Save the right camera calibration result for later use
pickle.dump((cameraMatrix_right, dist_right), open("right_calibration.pkl", "wb"))

# Print the calibration results (optional)
print("Left Camera Calibration Result:")
print("Camera Matrix:\n", cameraMatrix_left)
print("Distortion Coefficients:\n", dist_left)

print("Right Camera Calibration Result:")
print("Camera Matrix:\n", cameraMatrix_right)
print("Distortion Coefficients:\n", dist_right)


#############################################
# for single cam

# import numpy as np
# import cv2 as cv
# import glob
# import pickle



# ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

# chessboardSize = (7, 10)
# frameSize = (640,480)



# # termination criteria
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
# objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

# size_of_chessboard_squares_mm = 20
# objp = objp * size_of_chessboard_squares_mm


# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.


# images = glob.glob('images/*.png')

# for image in images:

#     img = cv.imread(image)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     # Find the chess board corners
#     ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

#     # If found, add object points, image points (after refining them)
#     if ret == True:

#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners)

#         # Draw and display the corners
#         cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(1000)

# print("Number of valid calibration images:", len(objpoints))

# cv.destroyAllWindows()




# ############## CALIBRATION #######################################################

# ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
# pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
# pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
# pickle.dump(dist, open( "dist.pkl", "wb" ))


# ############## UNDISTORTION #####################################################

# img = cv.imread('cali.png')
# h,  w = img.shape[:2]
# newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



# # Undistort
# dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('caliResult1.png', dst)



# # Undistort with Remapping
# mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('caliResult2.png', dst)




# # Reprojection Error
# mean_error = 0

# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     mean_error += error

# print( "total error: {}".format(mean_error/len(objpoints)) )



import numpy as np
import cv2 as cv
import glob
import pickle
import os

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9, 6)
frameSize = (640, 480)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints_left = []  # 3d point in real world space
imgpoints_left = []  # 2d points in image plane.

objpoints_right = []  # 3d point in real world space
imgpoints_right = []  # 2d points in image plane.

left_images = glob.glob('left_images/*.png')
right_images = glob.glob('right_images/*.png')

for left_image, right_image in zip(left_images, right_images):
    # Process left camera image
    img_left = cv.imread(left_image)
    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)

    # Find the chess board corners for the left camera image
    ret_left, corners_left = cv.findChessboardCorners(gray_left, chessboardSize, None)

    # If found, add object points, image points (after refining them) for the left camera
    if ret_left == True:
        objpoints_left.append(objp)
        corners2_left = cv.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        imgpoints_left.append(corners2_left)
        print("Left chessboard corners found.")
    else:
        print("Left chessboard corners not found.")

        # # Draw and display the corners for the left camera image
        # cv.drawChessboardCorners(img_left, chessboardSize, corners2_left, ret_left)
        # cv.imshow('Left Camera', img_left)
        # cv.waitKey(500)

    # Process right camera image
    img_right = cv.imread(right_image)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    # Find the chess board corners for the right camera image
    ret_right, corners_right = cv.findChessboardCorners(gray_right, chessboardSize, None)

    # If found, add object points, image points (after refining them) for the right camera
    if ret_right == True:
        objpoints_right.append(objp)
        corners2_right = cv.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        imgpoints_right.append(corners2_right)
        print("Right chessboard corners found.")
    else:
        print("Right chessboard corners not found.")

        # # Draw and display the corners for the right camera image
        # cv.drawChessboardCorners(img_right, chessboardSize, corners2_right, ret_right)
        # cv.imshow('Right Camera', img_right)
        # cv.waitKey(500)

cv.destroyAllWindows()

print("Number of valid left calibration images:", len(objpoints_left))
print("Number of valid right calibration images:", len(objpoints_right))

############## CALIBRATION #######################################################

# Calibrate left camera
ret_left, cameraMatrix_left, dist_left, rvecs_left, tvecs_left = cv.calibrateCamera(objpoints_left, imgpoints_left, frameSize, None, None)

# Save the left camera calibration result
os.makedirs('left_calibration', exist_ok=True)
pickle.dump((cameraMatrix_left, dist_left), open("left_calibration/calibration.pkl", "wb"))
pickle.dump(cameraMatrix_left, open("left_calibration/cameraMatrix.pkl", "wb"))
pickle.dump(dist_left, open("left_calibration/dist.pkl", "wb"))

# Calibrate right camera
ret_right, cameraMatrix_right, dist_right, rvecs_right, tvecs_right = cv.calibrateCamera(objpoints_right, imgpoints_right, frameSize, None, None)

# Save the right camera calibration result
os.makedirs('right_calibration', exist_ok=True)
pickle.dump((cameraMatrix_right, dist_right), open("right_calibration/calibration.pkl", "wb"))
pickle.dump(cameraMatrix_right, open("right_calibration/cameraMatrix.pkl", "wb"))
pickle.dump(dist_right, open("right_calibration/dist.pkl", "wb"))

############## UNDISTORTION #####################################################

# Undistort left camera image
img_left = cv.imread('cali.png')
h, w = img_left.shape[:2]
newCameraMatrix_left, roi_left = cv.getOptimalNewCameraMatrix(cameraMatrix_left, dist_left, (w, h), 1, (w, h))

dst_left = cv.undistort(img_left, cameraMatrix_left, dist_left, None, newCameraMatrix_left)
x, y, w, h = roi_left
dst_left = dst_left[y:y+h, x:x+w]
cv.imwrite('left_calibration/caliResult.png', dst_left)

# Undistort right camera image
img_right = cv.imread('cali.png')
h, w = img_right.shape[:2]
newCameraMatrix_right, roi_right = cv.getOptimalNewCameraMatrix(cameraMatrix_right, dist_right, (w, h), 1, (w, h))

dst_right = cv.undistort(img_right, cameraMatrix_right, dist_right, None, newCameraMatrix_right)
x, y, w, h = roi_right
dst_right = dst_right[y:y+h, x:x+w]
cv.imwrite('right_calibration/caliResult.png', dst_right)

############## REPROJECTION ERROR #####################################################

# Calculate reprojection error for left camera
mean_error_left = 0
for i in range(len(objpoints_left)):
    imgpoints2_left, _ = cv.projectPoints(objpoints_left[i], rvecs_left[i], tvecs_left[i], cameraMatrix_left, dist_left)
    error_left = cv.norm(imgpoints_left[i], imgpoints2_left, cv.NORM_L2) / len(imgpoints2_left)
    mean_error_left += error_left
print("Left camera total reprojection error: {}".format(mean_error_left / len(objpoints_left)))

# Calculate reprojection error for right camera
mean_error_right = 0
for i in range(len(objpoints_right)):
    imgpoints2_right, _ = cv.projectPoints(objpoints_right[i], rvecs_right[i], tvecs_right[i], cameraMatrix_right, dist_right)
    error_right = cv.norm(imgpoints_right[i], imgpoints2_right, cv.NORM_L2) / len(imgpoints2_right)
    mean_error_right += error_right
print("Right camera total reprojection error: {}".format(mean_error_right / len(objpoints_right)))

#############################################
# for single cam

# import numpy as np
# import cv2 as cv
# import glob
# import pickle



# ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

# chessboardSize = (9,6)
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

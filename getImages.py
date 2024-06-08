# import cv2

# #raspbery pi usb camera
# # cap = cv2.VideoCapture(4, cv2.CAP_V4L) # 0 is the camera index
# # cap = cv2.VideoCapture(2, cv2.CAP_V4L) # 2 is the camera index

# # windows usb camera

# # Try to initialize the camera with different indices and backends
# indices = [ 1,2]  # List of indices to try
# backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, None]  # List of backends to try

# cap = None
# for backend in backends:
#     for index in indices:
#         cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
#         if cap.isOpened():
#             print(f"Camera opened successfully with index {index} and backend {backend}")
#             break
#     if cap and cap.isOpened():
#         break

# num = 0

# if not cap or not cap.isOpened():
#     print("Error: Could not open any camera.")
#     exit()
# while True:
#     success, img = cap.read()

#     if not success:
#         print("Failed to capture image")
#         break

#     cv2.imshow('Img', img)

#     k = cv2.waitKey(5)

#     if k == 27:  # ESC key to exit
#         break
#     elif k == ord('s'):  # 's' key to save and exit
#         cv2.imwrite('left_images/img' + str(num) + '.png', img)
#         print("image saved!")
#         num += 1

# # Release and destroy all windows before termination
# cap.release()
# cv2.destroyAllWindows()

import cv2
import os
import platform

# Ensure the directories exist
os.makedirs('right_images', exist_ok=True)
os.makedirs('left_images', exist_ok=True)

# Determine the appropriate backend for the platform
current_platform = platform.system()
if current_platform == 'Windows':
    backend = cv2.CAP_DSHOW
else:
    backend = cv2.CAP_V4L2  # Video4Linux for Linux/Raspberry Pi

# Initialize the cameras
right_cam_index = 1
left_cam_index = 2

right_cap = cv2.VideoCapture(right_cam_index, backend)
left_cap = cv2.VideoCapture(left_cam_index, backend)

if not right_cap.isOpened():
    print(f"Error: Could not open right camera on index {right_cam_index}")
    exit()

if not left_cap.isOpened():
    print(f"Error: Could not open left camera on index {left_cam_index}")
    exit()

right_num = 0
left_num = 0

while True:
    right_success, right_img = right_cap.read()
    left_success, left_img = left_cap.read()

    if not right_success:
        print("Failed to capture image from right camera")
        break

    if not left_success:
        print("Failed to capture image from left camera")
        break

    cv2.imshow('Right Camera', right_img)
    cv2.imshow('Left Camera', left_img)

    k = cv2.waitKey(5)

    if k == 27:  # ESC key to exit
        break
    elif k == ord('r'):  # 'r' key to save image from right camera
        cv2.imwrite(f'right_images/img{right_num}.png', right_img)
        print("Right image saved!")
        right_num += 1
    elif k == ord('l'):  # 'l' key to save image from left camera
        cv2.imwrite(f'left_images/img{left_num}.png', left_img)
        print("Left image saved!")
        left_num += 1

# Release and destroy all windows before termination
right_cap.release()
left_cap.release()
cv2.destroyAllWindows()

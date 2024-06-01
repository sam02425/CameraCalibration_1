import cv2

# cap = cv2.VideoCapture(0, cv2.CAP_V4L) # 0 is the camera index
cap = cv2.VideoCapture(2, cv2.CAP_V4L) # 2 is the camera index


num = 0

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()


# import cv2
# import os

# # Create the 'left_images' and 'right_images' folders if they don't exist
# os.makedirs('left_images', exist_ok=True)
# os.makedirs('right_images', exist_ok=True)

# cap_left = cv2.VideoCapture(2, cv2.CAP_V4L)  # 2 is the left camera index
# cap_right = cv2.VideoCapture(0, cv2.CAP_V4L)  # 0 is the right camera index

# num = 0

# while cap_left.isOpened() and cap_right.isOpened():
#     success_left, img_left = cap_left.read()
#     success_right, img_right = cap_right.read()

#     if not success_left or not success_right:
#         print("Failed to grab frame from one or both cameras")
#         break

#     cv2.imshow('Left Camera', img_left)
#     cv2.imshow('Right Camera', img_right)

#     k = cv2.waitKey(5)

#     if k == 27:  # ESC key
#         break
#     elif k == ord('s'):  # 's' key to save images and exit
#         left_img_name = f'left_images/left_img{num}.png'
#         right_img_name = f'right_images/right_img{num}.png'

#         cv2.imwrite(left_img_name, img_left)
#         cv2.imwrite(right_img_name, img_right)

#         print(f"Images saved!")
#         print(f"Left image saved: {left_img_name}")
#         print(f"Right image saved: {right_img_name}")
#         print(f"Left image shape: {img_left.shape}")
#         print(f"Right image shape: {img_right.shape}")

#         num += 1

# # Release and destroy all windows before termination
# cap_left.release()
# cap_right.release()
# cv2.destroyAllWindows()

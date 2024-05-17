import cv2

cap = cv2.VideoCapture(2, cv2.CAP_V4L)  # 2 is the camera index

num = 0

while cap.isOpened():
    succes, img = cap.read()
    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'):  # Wait for 's' key to save and exit
        cv2.imwrite('images/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img', img)

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()

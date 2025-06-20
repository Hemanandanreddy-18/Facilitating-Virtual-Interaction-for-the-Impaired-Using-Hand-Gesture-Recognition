import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300

folder = "Data2/Where is the Restroom"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:

        combinedImgWhite = np.ones((imgSize, imgSize * 2, 3), np.uint8) * 255

        for i, hand in enumerate(hands):
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w


            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            if i == 0:
                combinedImgWhite[:, :imgSize] = imgWhite
            else:
                combinedImgWhite[:, imgSize:] = imgWhite


            cv2.imshow(f"ImageCrop_{hand['type']}", imgCrop)
            cv2.imshow(f"ImageWhite_{hand['type']}", imgWhite)


        cv2.imshow("Combined Image", combinedImgWhite)


    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1

        cv2.imwrite(f'{folder}/Image_combined_{time.time()}.jpg', combinedImgWhite)
        print(f"Image saved: {counter}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

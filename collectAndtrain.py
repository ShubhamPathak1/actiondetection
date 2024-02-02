# from importer import *
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PoseModule import PoseDetector
import numpy as np
import pyttsx3
import math
import time

capture = cv2.VideoCapture(0)
handDetector = HandDetector(maxHands=2)
facemeshDetector = FaceMeshDetector(maxFaces=1)
poseDetector = PoseDetector()

offset = 20
imgSize = 300

folder = "Data/Help"
counter = 0
capture_timer = time.time()

while True:
    success, img = capture.read()
    hands, img = handDetector.findHands(img)
    img, faces = facemeshDetector.findFaceMesh(img)
    img = poseDetector.findPose(img)
    lmlist, bboxInfo = poseDetector.findPosition(img)
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
    if lmlist:
        x,y,w,h=bboxInfo['bbox']
        # imgCrop = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        imgCrop = img[y:y+h,x:x+w]
        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap] = imgResize
            # speak(s"hands vertical")
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hGap+hCal, :] = imgResize
            # speak("hands horizontal")


        # aspectRatio = 

        # cv2.imshow("cropped", imgCrop)
        # cv2.imshow("white", imgWhite)
        
    cv2.imshow("Image", img)

    # if cv2.waitKey(1) & 0xFF==ord('s'):
    #     counter+=1
    #     cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
    #     print(counter)
    time.sleep(2)
    if time.time() - capture_timer >= 1:
        capture_timer = time.time()
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
# capture.release()
# cv2.destroyAllWindows()
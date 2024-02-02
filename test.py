import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import math
import numpy as np
import pyttsx3
from cvzone.ClassificationModule import Classifier
import tensorflow

def speak(tinp):
    engine = pyttsx3.init()
    engine.say(tinp)
    engine.runAndWait()

cam = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

folder = "Images/C"
counter = 0

labels = ["A", "B" ,"C"]

while True:
    success, img = cam.read()
    imgOutput = img.copy()
    hands= detector.findHands(img, draw=False)
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
    for hand in hands:

        x, y, w, h = hand['bbox']

        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap] = imgResize
            # speak("hands vertical")
            prediction, index = classifier.getPrediction(img)
            print(prediction, index)
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hGap+hCal, :] = imgResize
            # speak("hands horizontal")
            classifier.getPrediction(img)
            print(prediction, index)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

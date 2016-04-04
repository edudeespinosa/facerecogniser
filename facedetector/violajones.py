# import the necessary packages
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import facedetector as fdlib
import cv2

class ViolaJones:
    def __init__(self, imagePath):
        self.face_cascade = cv2.CascadeClassifier('./facedetector/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('./facedetector/haarcascade_eye.xml')

        self.imageToDetect = cv2.imread(imagePath)

    def vj(self):
        gray = cv2.cvtColor(self.imageToDetect, cv2.COLOR_BGR2GRAY)
        fd = fdlib.FaceDetector('./facedetector/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(self.imageToDetect, cv2.COLOR_BGR2GRAY)
        faces = fd.detect(gray, scaleFactor = 1.08, minNeighbors = 5,
                          minSize = (30, 30))
        print(faces)
        for (x,y,w,h) in faces:
            print("face")
            cv2.rectangle(self.imageToDetect,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = self.imageToDetect[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            cv2.rectangle(self.imageToDetect,(x,y),(x+w,y+h),(255,0,0),2)


        cv2.imshow('Image', self.imageToDetect)
        cv2.waitKey(0)

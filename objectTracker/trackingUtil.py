import sys
from datetime import datetime
import cv2
import numpy as np
import Image
import faceDetector.violajones as vj
from faceRecogniser.facerecutil import read_training_images as reader
from faceRecogniser.model import EigenfacesModel
import faceRecogniser.facerecutil as util
import faceRecogniser.pca as pca


def faceInWindow(faceNumber, image):
    name = "face "+str(faceNumber)
    # cv2.imshow(name, image)
    # cv2.resizeWindow(name, 300, 200)
    # cv2.moveWindow(name, 0, 200*faceNumber-1)


def saveImage(faceNumber, image):
    imgname = "./recovered/"+str(datetime.now())+str(faceNumber)+".png"
    cv2.imwrite(imgname, image)
    return imgname


def faceFinding(roiPts, frame, model, sz, filenames):
    orig = frame.copy()
    violajones = vj.ViolaJones(orig, 'Video')
    faceRects = violajones.vj()
    print("Total faces : "+str(len(faceRects)))
    count = 1
    for (x, y, w, h) in faceRects:
        x = (x+10)
        y = (y+10)
        w = (w-15)
        h = (h-15)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roiPts.append((x, y, x+w, y+h))

        extra_img = orig[y: y+h, x: x+w]
        imgname = saveImage(count, extra_img)

        im = Image.open(imgname)
        im = im.convert("L")
        # resize to given size (if given)
        im = im.resize(sz, Image.ANTIALIAS)
        im = np.asarray(im, dtype=np.uint8)
        print("Predicting")
        prediction = model.predict(im)
        subject = filenames[prediction]
        print(prediction)
        alpha = 0.3
        if (subject == 'Eduardo'):
            copy = extra_img.copy()
            cv2.rectangle(
                extra_img, (0, 0), (w, h), (0, 255, 0), cv2.cv.CV_FILLED
            )
            dst = cv2.addWeighted(extra_img,0.3,copy,0.7,0)
            cv2.imshow("Identified Person "+str(subject), dst)
        print("Predicted subject: "+str(subject))


def createHist(roiPts, frame, objectsToTrack, roiHist):
    i = 1
    orig = frame.copy()
    for rp in roiPts:
        # Grab the ROI for the bounding box by cropping and convert it
        # to the HSV color space.
        roi = orig[rp[1]:rp[-1], rp[0]:rp[2]]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # compute a HSV histogram for the ROI and store the
        # bounding box
        rh = cv2.calcHist([roi], [0], None, [16], [0, 180])
        rh = cv2.normalize(rh, rh, 0, 255, cv2.NORM_MINMAX)
        objectsToTrack.append((rp[0], rp[1], rp[2], rp[-1]))
        roiHist.append(rh)
        i = i+1


def initFaceRecognitionModel(dataset):
    if(dataset == 'Yale'):
        sz = [320, 240]
        [X, y, filenames] = reader("./faceRecogniser/yalefaces", sz)
    elif (dataset == 'ATT'):
        sz = [92, 112]
        [X, y, filenames] = reader("./faceRecogniser/faces", sz)

    for i in range(0, len(filenames)):
        filenames[i] = filenames[i].split("faces/s")[1]

    print("Reading training faces")

    print("Generating model")
    model = EigenfacesModel(X[1:], y[1:])

    [D, W, mu] = model.compute(X[1:], y[1:])

    E = []

    return (sz, model, filenames)

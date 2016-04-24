from datetime import datetime
import cv2
import faceDetector.violajones as vj


def faceInWindow(faceNumber, image):
    name = "face "+str(faceNumber)
    cv2.imshow(name, image)
    cv2.resizeWindow(name, 300, 200)
    cv2.moveWindow(name, 0, 200*faceNumber-1)


def saveImage(faceNumber, image):
    imgname = "./recovered/"+str(datetime.now())+str(faceNumber)+".png"
    cv2.imwrite(imgname, image)


def faceFinding(roiPts, frame):
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

        saveImage(count, extra_img)

        count = count+1

        # append points from violajones
        # show the detected faces


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

import numpy as np
import cv2
import faceDetector.faceDetector as fd
import trackingUtil as util
roiPts = []
SKIP = 30


class Tracker:
    global roiPts
    global SKIP

    def __init__(self, source, face):
        self.frame = None
        self.source = source
        if source == None:
            self.type = 'camera'
        else:
            self.type = 'video'
        self.objectsToTrack = []
        self.face = face
        self.roiHist = []
        if self.face != None:
            (self.sz, self.model, self.filenames) = util.initFaceRecognitionModel('ATT')
        self.termCrit = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # Seleccionar area de interes en el self.frame deseado.
    def boxSelection(self, event, x, y, flags, param):
        global roiPts
        # recibir los clicks del left button
        if event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
            RoiPoint = (x, y)
            roiPts.append(RoiPoint)
            cv2.circle(self.frame, RoiPoint, 10, (255, 255, 255), 2)
            cv2.imshow("frame", self.frame)

    def trackObject(self):
        global roiPts
        global SKIP
        if self.type == 'camera':
            camera = cv2.VideoCapture(0)
        else:
            camera = cv2.VideoCapture(self.source)
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", self.boxSelection)
        i = 0
        skip = 0
        while camera.read():
            # seleccionar self.frame
            (selected, self.frame) = camera.read()

            # existe video?
            if not selected:
                print("really?")
                break

            # aqui hay dos opciones:
            #   ya hay objeto para seguir. Es decir, ya hay area de interese seleccionada
            # No hay objeto seleccionado para seguir. Se debe entrar al modo de
            # seleccion

            # ya hay objeto?
            if self.objectsToTrack:
                skip = skip + 1
                count = 1
                for objectToTrack in self.objectsToTrack:
                    # convertir a hsv
                    hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                    backProj = cv2.calcBackProject(
                        [hsv], [0], self.roiHist[count-1], [0, 180], 1)

                    color = (255, 0, 0)
                    # aplicar camshift/meanshift
                    if self.face == None:
                        (r, objectToTrack) = cv2.meanShift(
                            backProj, objectToTrack, self.termCrit)
                        x, y, w, h = objectToTrack

                        cv2.rectangle(self.frame, (x, y), (x+w, y+h), color, 2)

                    else:
                        (r, objectToTrack) = cv2.CamShift(
                            backProj, objectToTrack, self.termCrit)

                        # To draw polygones inside frame:
                        pts = np.int0(cv2.cv.BoxPoints(r))
                        cv2.polylines(self.frame, [pts], True, color, 1)
                        x, y, w, h = objectToTrack

                        extra_img = self.frame[y: y+h, x: x+w]
                        util.faceInWindow(count, extra_img)

                    count = count+1

                if self.face != None:
                    del roiPts[:]

            # Esperar a que se presione tecla
            if(self.face == None):
                key = cv2.waitKey(1) & 0xFF
                self.manuallySelectObject(key)
            else:
                cv2.waitKey(1) & 0xFF
                # Every 30 frames, look for new faces
                if not self.objectsToTrack or skip == SKIP:
                    self.objectsToTrack = []
                    util.faceFinding(roiPts, self.frame, self.model, self.sz, self.filenames)
                    util.createHist(
                        roiPts, self.frame, self.objectsToTrack, self.roiHist)
                    skip = 0
                # if (len(roiPts)!=0):
                #     self.createHist()
            cv2.imshow("frame", self.frame)

        camera.release()
        cv2.destroyAllWindows()

    def manuallySelectObject(self, key):
        global roiPts
        if key == ord("s") and len(roiPts) < 4:
            orig = self.frame.copy()

            # seleccionar bounding box para el objeto a seguir
            while len(roiPts) < 4:
                cv2.imshow("frame", self.frame)
                cv2.waitKey(0)

            # Escoger puntos arriba y abajo.
            roiPts = np.array(roiPts)
            s = roiPts.sum(axis=1)
            tl = roiPts[np.argmin(s)]
            br = roiPts[np.argmax(s)]

            # transformar a hsv
            roi = orig[tl[1]:br[1], tl[0]:br[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # obtener histograma de colores y normalizar
            rh = cv2.calcHist([roi], [0], None, [16], [0, 180])
            self.roiHist.append(cv2.normalize(rh, rh, 0, 255, cv2.NORM_MINMAX))
            self.objectsToTrack.append((tl[0], tl[1], br[0], br[1]))

        # salir del programa
        elif key == ord("q"):
            return quit()

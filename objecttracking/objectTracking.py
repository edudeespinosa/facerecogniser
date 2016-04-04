import numpy as np
import argparse
import cv2

roiPts = []

class Tracker:
    global roiPts
    def __init__(self, source, face):
        self.frame = None
        self.source = source
        if source == None:
            self.type = 'camera'
        else:
            self.type= 'video'
        self.objectToTrack = None
        self.face = face
        self.roiHist = None
    # Seleccionar area de interes en el self.frame deseado.
    def boxSelection(self, event, x, y, flags, param):
        global roiPts
        # recibir los clicks del left button
        if event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
            RoiPoint = (x,y)
            roiPts.append(RoiPoint)
            cv2.circle(self.frame, RoiPoint, 10, (255, 255, 255), 2)
            cv2.imshow("frame", self.frame)
    def trackObject(self):
        if self.type == 'camera':
            camera = cv2.VideoCapture(0)
        else:
            camera = cv2.VideoCapture(self.source)
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", self.boxSelection)

        termCrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        while True:
            # seleccionar self.frame
            (selected, self.frame) = camera.read()


            # existe video?
            if not selected:
                break


            # aqui hay dos opciones:
            #   ya hay objeto para seguir. Es decir, ya hay area de interese seleccionada
            #   No hay objeto seleccionado para seguir. Se debe entrar al modo de seleccion

            # ya hay objeto?
            if self.objectToTrack is not None:
                # convertir a hsv
                hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                backProj = cv2.calcBackProject([hsv], [0], self.roiHist, [0, 180], 1)

                # aplicar camshift
                if self.face==None:
                    (r, self.objectToTrack) = cv2.meanShift(backProj, self.objectToTrack, termCrit)
                    x,y,w,h = self.objectToTrack
                    cv2.rectangle(self.frame, (x,y), (x+w,y+h), 255,2)
                else:
                    (r, self.objectToTrack) = cv2.CamShift(backProj, self.objectToTrack, termCrit)
                    pts = np.int0(cv2.cv.BoxPoints(r))
                    cv2.polylines(self.frame, [pts], True, (0, 255, 0), 2)

            cv2.imshow("frame", self.frame)

            # Esperar a que se presione tecla
            key = cv2.waitKey(1) & 0xFF
            if ( self.face == None ):
                self.manuallySelectObject(key)
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
            s = roiPts.sum(axis = 1)
            tl = roiPts[np.argmin(s)]
            br = roiPts[np.argmax(s)]

            # transformar a hsv
            roi = orig[tl[1]:br[1], tl[0]:br[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # obtener histograma de colores y normalizar
            self.roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
            self.roiHist = cv2.normalize(self.roiHist, self.roiHist, 0, 255, cv2.NORM_MINMAX)
            self.objectToTrack = (tl[0], tl[1], br[0], br[1])

        # salir del programa
        elif key == ord("q"):
            return quit()


from datetime import datetime
import numpy as np
import cv2
import faceDetector.violajones as vj
import faceDetector.faceDetector as fd
roiPts = []
tracking = False

class Tracker:
    global roiPts
    def __init__(self, source, face):
        self.frame = None
        self.source = source
        if source == None:
            self.type = 'camera'
        else:
            self.type= 'video'
        self.objectsToTrack = []
        self.face = face
        self.roiHist = []
        self.termCrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

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
        global roiPts
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
            #   No hay objeto seleccionado para seguir. Se debe entrar al modo de seleccion


            # ya hay objeto?
            if self.objectsToTrack:
                skip = skip + 1
                tracking = True
                count = 1
                for objectToTrack in self.objectsToTrack:
                    # convertir a hsv
                    hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                    backProj = cv2.calcBackProject([hsv], [0], self.roiHist[count-1], [0, 180], 1)

                    # aplicar camshift/meanshift
                    if self.face==None:
                        color = (255,0,0)
                        (r, objectToTrack) = cv2.meanShift(backProj, objectToTrack, self.termCrit)
                        x,y,w,h = objectToTrack
                        x = (x+10)
                        y = (y+10)
                        w = (w-15)
                        h = (h-15)
                        cv2.rectangle(self.frame, (x,y), (x+w,y+h), color,2)

                    else:
                        color = (255,255,0)
                        (r, objectToTrack) = cv2.CamShift(backProj, objectToTrack, self.termCrit)

                        pts = np.int0(cv2.cv.BoxPoints(r))        
                        # cv2.polylines(self.frame, [pts], True, color, 1)
                        x,y,w,h = objectToTrack
                        x = (x+10)
                        y = (y+10)
                        w = (w-15)
                        h = (h-15)

                        extra_img = self.frame[y: y+h, x: x+w]
                        self.faceInWindow(count, extra_img)

                    count = count+1
                
                if self.face != None:
                    # del self.objectsToTrack[:]
                    del roiPts[:]

            cv2.imshow("frame", self.frame)
            # Esperar a que se presione tecla
            if(self.face==None):
                key = cv2.waitKey(1) & 0xFF
                self.manuallySelectObject(key)
            else:
                cv2.waitKey(1) & 0xFF
                if not self.objectsToTrack or skip == 30:
                    self.faceFinding()
                    skip = 0
                # if (len(roiPts)!=0):
                #     self.createHist()


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
            rh = cv2.calcHist([roi], [0], None, [16], [0, 180])
            self.roiHist.append(cv2.normalize(rh, rh, 0, 255, cv2.NORM_MINMAX))
            self.objectsToTrack.append((tl[0], tl[1], br[0], br[1]))

        # salir del programa
        elif key == ord("q"):
            return quit()

    def faceFinding(self):
        global roiPts
        self.objectsToTrack = []
        orig = self.frame.copy()
        violajones = vj.ViolaJones(self.frame, 'Video')
        faceRects = violajones.vj()
        # dim = (self.frame.shape[1], self.frame.shape[0]);
        # #Aplicar Viola jones. Transformar a gris
        # gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (10, 10))
        print("Total faces : "+str(len(faceRects)))
        count = 1
        for (x, y, w, h) in faceRects:
            x = (x+10)
            y = (y+10)
            w = (w-15)
            h = (h-15)
            # cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roiPts.append((x, y, x+w, y+h))
            
            extra_img = self.frame[y: y+h, x: x+w]
            imgname = "./recovered/"+str(datetime.now())+str(count)+".png"
            cv2.imwrite(imgname, extra_img)


            count = count+1

            ## append points from violajones
            #show the detected faces
        self.createHist()

    def faceInWindow(self, faceNumber, image):
        name = "face "+str(faceNumber)
        cv2.imshow(name, image)
        cv2.resizeWindow(name, 300, 200)
        cv2.moveWindow(name, 0, 200*faceNumber-1)
        


    def createHist(self):
        global roiPts
        i = 1
        orig = self.frame.copy()
        for rp in roiPts:
            # Grab the ROI for the bounding box by cropping and convert it
            # to the HSV color space.
            roi = orig[rp[1]:rp[-1], rp[0]:rp[2]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # compute a HSV histogram for the ROI and store the
            # bounding box
            rh = cv2.calcHist([roi], [0], None, [16], [0, 180])
            rh = cv2.normalize(rh, rh, 0, 255, cv2.NORM_MINMAX)
            self.objectsToTrack.append((rp[0], rp[1], rp[2], rp[-1]))
            self.roiHist.append(rh)
            i = i+1

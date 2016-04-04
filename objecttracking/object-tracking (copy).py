import numpy as np
import argparse
import cv2

frame = None
roiPts = []




# Seleccionar area de interes en el frame deseado.
def boxSelection(event, x, y, flags, param):
    global frame, roiPts
    # recibir los clicks del left button
    if event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        RoiPoint = (x,y)
        roiPts.append(RoiPoint)
        cv2.circle(frame, RoiPoint, 10, (255, 255, 255), 2)
        cv2.imshow("frame", frame)

def main():
    global frame, roiPts

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
        help = "Ruta del video")
    args = vars(ap.parse_args())


    # recibir camara de video. CV2 permite tambien recibir desde un archivo de video
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])


    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", boxSelection)

    termCrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    objectToTrack = None

    while True:
        # seleccionar frame
        (selected, frame) = camera.read()


        # existe video?
        if not selected:
            break


        # aqui hay dos opciones:
        #   ya hay objeto para seguir. Es decir, ya hay area de interese seleccionada
        #   No hay objeto seleccionado para seguir. Se debe entrar al modo de seleccion

        # ya hay objeto?
        if objectToTrack is not None:
            # convertir a hsv
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)

            # aplicar camshift
            (r, objectToTrack) = cv2.CamShift(backProj, objectToTrack, termCrit)
            pts = np.int0(cv2.cv.BoxPoints(r))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        cv2.imshow("frame", frame)

        # Esperar a que se presione tecla
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and len(roiPts) < 4:
            orig = frame.copy()

            # seleccionar bounding box para el objeto a seguir
            while len(roiPts) < 4:
                cv2.imshow("frame", frame)
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
            roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
            objectToTrack = (tl[0], tl[1], br[0], br[1])

        # salir del programa
        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
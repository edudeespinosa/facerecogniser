import objecttracking.objectTracking as ot
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help = "Ruta del video")
args = vars(ap.parse_args())


# recibir camara de video. CV2 permite tambien recibir desde un archivo de video
if not args.get("video", False):
    faceTracker = ot.Tracker(None, True)
else:
    faceTracker = ot.Tracker(args["video"], True)

faceTracker.trackObject()
# import the necessary packages
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import argparse
import facedetector.violajones as vj

global image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
    help = "Ruta de la imagen")
args = vars(ap.parse_args())

if not args.get("image", False):
    image = './facedetector/faces.jpg'
else:
    image = (args["image"])


violajones = vj.ViolaJones(image)
violajones.vj()



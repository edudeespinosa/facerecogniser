import sys
from faceRecogniser.facerecutil import read_training_images as reader
from faceRecogniser.model import EigenfacesModel
import faceRecogniser.facerecutil as util
import faceRecogniser.pca as pca
from random import shuffle, randint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
import os
import Image
import cv2

[X,y] = reader("./faceRecogniser/yalefaces")

model = EigenfacesModel(X[1:], y[1:])

[D, W, mu]= model.compute(X[1:], y[1:])

copy = X

shuffle(copy)
E=[]
for i in xrange(0,16):
    e = W[:, i].reshape(X[0].shape)
    image = util.normalize(e, 0, 255)
    # cv2.imshow(str(i), image)
    E.append(util.normalize(e, 0, 255))


filename="./pca/python_pca_eigenfaces.png"
pca.subplot(title=" Eigenfaces Yale ", images=E, rows=4, cols=4,
        sptitle="Subject ", colormap=cm.jet,
        filename=filename)


img = mpimg.imread(filename)
# plt.imshow(np.asarray(img))
# plt.show()
# raw_input("Waiting")
# plt.clf()

for i in range(0, len(X)):
    prediction = model.predict(copy[i])
    print("Predicted subject: "+str(prediction))
    cv2.imshow("Subject "+str(model.predict(copy[i])), copy[i])
    # plt.imshow(E[prediction], interpolation='nearest')
    # plt.show()
    cv2.waitKey(1) & 0xFF
    # print ("expected: ", y[i], "/", "predicted:", model.predict(X[i]))
raw_input("Waiting")

fig = plt.figure()

for i in xrange(0,10):
    random = randint(0, 159)
    prediction = model.predict((copy[random]))
    print("Predicted subject: "+str(prediction))
    ax0 = fig.add_subplot(10, 2, ((i*2)+1))
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax0.get_yticklabels(), visible=False)
    plt.title("image:")
    plt.imshow((copy[random]))
    ax0 = fig.add_subplot(10, 2, ((i*2)+2))
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax0.get_yticklabels(), visible=False)
    plt.title("Predicted subject: "+str(prediction))
    plt.imshow(E[prediction])

plt.show()
#
# for i in range(0, 10):
#     random = randint(0, 159)
#     prediction = model.predict((copy[random]))
#     print("Predicted subject: "+str(prediction))
#     cv2.imshow("Subject "+str(model.predict(copy[random])), copy[random])
#     plt.imshow(copy[prediction], interpolation='nearest')
#     plt.show()
#     raw_input("Waiting")
#

cv2.waitKey(1) & 0xFF


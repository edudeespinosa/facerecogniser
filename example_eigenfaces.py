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
print("Reading training faces")
# uncomment to use Yale faces
# [X,y, filenames] = reader("./faceRecogniser/yalefaces", [320, 240])


# uncomment to use ATT faces
[X, y, filenames] = reader("./faceRecogniser/faces", [92, 112])

print("Generating model")
model = EigenfacesModel(X[1:], y[1:])

[D, W, mu] = model.compute(X[1:], y[1:])

copy = X

# shuffle(copy)
E = []


# For Yale faces
# for i in xrange(0,16):

# For ATT faces
for i in xrange(0, 42):
    e = W[:, i].reshape(X[0].shape)
    image = util.normalize(e, 0, 255)
    # cv2.imshow(str(i), image)
    E.append(image)


filename = "./pca/python_pca_eigenfaces.png"


print(
    "We will print which eigenface is the nearest neighbor " +
    "between the training images and 10 random query images")
raw_input("press any key to continue")


fig = plt.figure()

for i in range(0, len(filenames)):
    filenames[i] = filenames[i].split("faces/s")[1]


# Show eigenfaces
# # For Yale faces
# pca.subplot(title=" Eigenfaces Yale ", images=E, rows=4, cols=4,

# For ATT faces
# # pca.subplot(title=" Eigenfaces ATT ", images=E, rows=9, cols=5,
#         sptitle="Subject ", colormap=cm.jet,
#         filename=filename)

shuffle(X)
for i in xrange(0, 10):
    random = randint(0, len(X))
    # while (model.predict(X[random])!= 23):
    # random = randint(0, len(X))
    prediction = model.predict(X[random])
    subject = filenames[prediction]
    ax0 = fig.add_subplot(10, 2, ((i*2)+1))
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax0.get_yticklabels(), visible=False)
    plt.title("Predicted subject: "+str(subject))
    # plt.title("predicted: "+str(prediction))
    plt.imshow((X[random]))
    ax0 = fig.add_subplot(10, 2, ((i*2)+2))
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax0.get_yticklabels(), visible=False)
    plt.title("Predicted subject: "+str(subject))
    print("Predicted subject: "+str(subject))
    print("Predicted folder: "+str(subject))
    plt.imshow(E[prediction])


plt.show()

shuffle(X)
print(
    "Now, we will shuffle the set of images and we will " +
    "predict the subject based on the face")
raw_input("Press any key to continue")
for i in range(0, len(X)):
    prediction = model.predict(copy[i])
    subject = filenames[prediction]
    print("Predicted subject: "+str(subject))
    cv2.imshow("Subject "+str(subject), copy[i])
    # plt.imshow(E[prediction], interpolation='nearest')
    # plt.show()
    cv2.waitKey(1) & 0xFF
    # print ("expected: ", y[i], "/", "predicted:", model.predict(X[i]))

raw_input("press any key to continue")
cv2.destroyAllWindows()


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

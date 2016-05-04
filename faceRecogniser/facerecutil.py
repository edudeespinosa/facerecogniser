import sys
import numpy as np
import os
import Image


def read_training_images(path, sz=None):
    c = 0
    X, y, files = [], [], []
    count = 0
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            print("subject "+str(c)+": "+subject_path)
            count = count+1
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as (errno, strerror):
                    print("I/O error({0}): {1}".format(errno, strerror))
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            files.append(subject_path)
            c = c+1
    return [X, y, files]


def asRowMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((0, X[0].size), dtype=X[0].dtype)
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1, -1)))
    return mat


def asColumnMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((X[0].size, 0), dtype=X[0].dtype)
    for col in X:
        mat = np.hstack((mat, np.asarray(col).reshape(-1, 1)))
    return mat


def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [ low...high ].
    X = X * (high - low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


class AbstractDistance (object):

    def __init__(self, name):
        self . _name = name

        def __call__(self, p, q):
            raise NotImplementedError(
                " Every AbstractDistance must implement the __call__method .")

    @property
    def name(self):
        return self . _name

    def __repr__(self):
        return self . _name


class EuclideanDistance (AbstractDistance):

    def __init__(self):
        AbstractDistance . __init__(self, " EuclideanDistance ")

    def __call__(self, p, q):
        p = np . asarray(p). flatten()
        q = np . asarray(q). flatten()
        return np . sqrt(np . sum(np . power((p - q), 2)))


class CosineDistance (AbstractDistance):

    def __init__(self):
        AbstractDistance . __init__(self, " CosineDistance ")

    def __call__(self, p, q):
        p = np . asarray(p). flatten()
        q = np . asarray(q). flatten()
        return -np . dot(p.T, q) / (np . sqrt(np . dot(p, p.T) * np . dot(q, q.T)))

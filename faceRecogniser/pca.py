import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def pca(X, y, num_components=0):
    [n, d] = X.shape
    if (num_components <= 0) or (num_components > n):
        num_components = n
    mu = X.mean(axis=0)
    X = X - mu
    if n > d:
        C = np.dot(X.T, X)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X, X.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in xrange(n):
            eigenvectors[:, i] = eigenvectors[
                :, i]/np.linalg.norm(eigenvectors[:, i])
    idx = np.argsort(- eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[0: num_components].copy()
    eigenvectors = eigenvectors[:, 0: num_components].copy()
    return [eigenvalues, eigenvectors, mu]


def project(W, X, mu=None):
    if mu is None:
        return np.dot(X, W)
    return np.dot(X - mu, W)


def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y, W.T)
    return np.dot(Y, W.T) + mu


def subplot(title, images, rows, cols, sptitle=" subplot ", sptitles=[],
            colormap=cm.gray, ticks_visible=True, filename=None):
    fig = plt.figure()
    # main title
    fig.text(.5, .95, title, horizontalalignment='center')
    for i in xrange(len(images)):
        ax0 = fig.add_subplot(rows, cols, (i + 1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])))
        else:
            plt.title("%s #%d" % (sptitle, (i + 1)))
            plt.imshow(np.asarray(images[i]), cmap=colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)

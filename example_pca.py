import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import faceRecogniser.pca as pcalib
import faceRecogniser.facerecutil as util
import matplotlib.cm as cm


# read images
[X, y] = util.read_training_images("./faceRecogniser/faces")
# perform a full pca
[D, W, mu] = pcalib.pca(util.asRowMatrix(X), y)
print(X)
# turn the first (at most ) 16 eigenvectors into grayscale
# images ( note : eigenvectors are stored by column !)
E = []
for i in xrange(min(len(X), 20)):
    e = W[:, i].reshape(X[0].shape)
    E.append(util.normalize(e, 0, 255))
# plot them and store the plot to " python_eigenfaces.pdf"
pcalib.subplot(title=" Eigenfaces AT&T Facedatabase ", images=E, rows=5, cols=4,
               sptitle="Eigenface ", colormap=cm.jet,
               filename="pca/python_pca_eigenfaces.png")

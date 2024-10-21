import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels

    def fit(self, xtr, ytr):
        """_summary_

        Args:
            xtr (_type_): _description_
            ytr (_type_): _description_

        Returns:
            _type_: _description_
        """

        n_dimensions = xtr.shape[1]
        n_classes = np.unique(ytr).size

        self._centroids = np.zeros((n_classes, n_dimensions))

        for i in range(0,n_classes):
            self._centroids[i,:]=np.mean(xtr[ytr==i,:],axis=0) #axis = dimension to remove
            #plt.subplot(2,5,i+1)
            #plt.imshow(centroids[i,:].reshape((28,28)))

        return self


    def predict(self, xts):
        pass

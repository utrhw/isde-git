import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def softmax(x):
    """_returns softmax of input array x_

    Args:
        x (_float_): _description_

    Returns:
        _float_: _description_
    """
    e_x = np.exp(x-np.max(x))
    return e_x/e_x.sum(axis=1, keepdims=True)


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
        pass

    def predict(self, xts):
        pass

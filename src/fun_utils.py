from pandas import read_csv
import numpy as np
from matplotlib import pyplot as plt


def split_data(X, y, tr_fraction=0.5):
    """
    Split the data X,y into two random subsets

    """
    num_samples = y.size
    n_tr = int(num_samples * tr_fraction)

    idx = np.array(range(0, num_samples))
    np.random.shuffle(idx)  # shuffle the elements of idx

    tr_idx = idx[0:n_tr]
    ts_idx = idx[n_tr:]

    Xtr = X[tr_idx, :]
    ytr = y[tr_idx]

    Xts = X[ts_idx, :]
    yts = y[ts_idx]

    return Xtr, ytr, Xts, yts


def plot_image(x, title, h=28, w=28):
    """
    Plots an image along with a title

    Parameters
    ----------
    x: the image given as a flat vector of size: (h*w, )
    title: the title to place on top of the image
    h: the image height
    w: the image width

    Returns
    -------
    None.
    """
    plt.imshow(x.reshape((h, w)), cmap=plt.cm.gray)
    # cmap = plt.cm.gray -> sets the colormap to grayscale values

    plt.title(str(title))


def compute_ts_error(ypred, yts):
    """
    Compute the fraction of elements that are different in ypred and yts
    (classification errors)

    Parameters
    ----------
    ypred: the set of predicted class labels
    yts: the true labels of test samples

    Returns
    -------
    test_error: the classification error
    """
    test_error = np.sum(ypred != yts) / float(ypred.size)
    print("Test error (" "on ", yts.size, " test samples): ", test_error)
    return test_error


def plot_ten_images(x, w, h, titles):
    """
    Display ten images given as rows in x in a 2x5 subplot,
    along with their titles (passed as a list of 10 strings)
    """
    for i in range(10):
        plt.subplot(2, 5, i + 1)  # select current subplot
        # display image inside the current plot
        plot_image(x[i, :], title=titles[i], w=w, h=h)


def count_samples_per_class(y):
    """
    Count the number of elements in each class

    Parameters
    ----------
    y : ndarray
        the labels of each sample.

    Returns
    -------
    v : ndarray
        the number of elements in each class.
    """
    classes = np.unique(y)
    num_classes = classes.size  # number of unique elements in y
    p = np.zeros(shape=(num_classes,))

    for k in range(num_classes):
        p[k] = np.sum(y == classes[k])

    return p


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y

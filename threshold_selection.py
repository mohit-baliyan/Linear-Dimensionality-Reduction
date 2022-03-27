import numpy as np


def threshold_selection(z, y):
    # total number of examples
    n = np.size(z)

    # reshape y into 1 D as of z
    # y = np.reshape(y, n)

    # define set of unique values with all possible points
    threshold = np.unique(z)

    # add all borders
    threshold = (threshold[1:] + threshold[:-1]) / 2

    # selecting threshold with best error
    best_error = n
    best_threshold = 0

    for t in threshold:

        y_hat = np.where(z > t, 1, 0)
        error = sum(y_hat != y)

        if error < best_error:
            best_error = error
            best_threshold = t

    return best_threshold

import numpy as np
from scipy.stats import mode


# K Nearest Neighbors Classification
class KNN:

    def __init__(self, k):
        self.k = k

    # function to store training set
    def fit(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train

        # no_of_training_examples, no_of_features
        self.m, self.n = self.X_train.shape

    # function for prediction
    def predict(self, X_test):

        # no_of_test_examples, no_of_features
        m_test, n = X_test.shape

        # initialize Y_predict
        y_predict = np.zeros(m_test)

        for i in range(m_test):

            X = X_test[i]

            # find the k nearest neighbors from current test example
            neighbors = np.zeros(self.k)
            neighbors = self.find_neighbors(X)

            # most frequent class in k neighbors
            y_predict[i] = mode(neighbors)[0][0]

        return y_predict

    # function to find the k nearest neighbors to current test example
    def find_neighbors(self, X):

        # calculate all the euclidean distances between current test example x and training set X_train
        euclidean_distances = np.zeros(self.m)
        X = np.reshape(X, (1, self.n))

        # create x for broadcasting with X_train
        X = np.repeat(X, repeats=self.m, axis=0)

        # (X - Y)^2 = X^2 + Y^2 - 2 * X * Y
        euclidean_distances = np.sum((np.square(X) + np.square(self.X_train) - 2 * self.X_train * X), axis=1)

        # sort y_train according to euclidean_distance_array and store into y_train_sorted
        ids = euclidean_distances.argsort()

        return self.y_train[ids[:self.k]]
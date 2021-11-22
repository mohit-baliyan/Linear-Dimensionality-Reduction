import numpy as np
from scipy.stats import mode


# K Nearest Neighbors Classification
class KNN:

    def __init__(self, k):
        self.k = k

    # Function to store training set
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        # no_of_training_examples, no_of_features
        self.m, self.n = self.X_train.shape

    # Function for prediction
    def predict(self, X_test):
        # no_of_test_examples, no_of_features
        m_test, n = X_test.shape
        # initialize Y_predict
        Y_predict = np.zeros(m_test)
        for i in range(m_test):
            x = X_test[i]
            # find the k nearest neighbors from current test example
            neighbors = np.zeros(self.k)
            neighbors = self.find_neighbors(x)
            # most frequent class in k neighbors
            Y_predict[i] = mode(neighbors)[0][0]
        return Y_predict

    # Function to find the k nearest neighbors to current test example
    def find_neighbors(self, x):
        # calculate all the euclidean distances between current test example x and training set X_train
        euclidean_distances = np.zeros(self.m)
        x = np.reshape(x, (1, self.n))
        # create x for broadcasting with X_train
        x = np.repeat(x, repeats=self.m, axis=0)
        # (X - Y)^2 = X^2 + Y^2 - 2 * X * Y
        euclidean_distances = np.sum((np.square(x) + np.square(self.X_train) - 2 * self.X_train * x), axis=1)
        # sort Y_train according to euclidean_distance_array and store into Y_train_sorted
        ids = euclidean_distances.argsort()
        return self.Y_train[ids[:self.k]]

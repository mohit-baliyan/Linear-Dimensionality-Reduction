import numpy as np
from scipy.stats import mode


# K Nearest Neighbors Classification
class KNN:

    def __init__(self, k):
        self.n = None
        self.m = None
        self.y_train = None
        self.x_train = None
        self.k = k

    # function to store training set
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        # no_of_training_examples, no_of_features
        self.m, self.n = self.x_train.shape

    # function for prediction
    def predict(self, x_test):
        # no_of_test_examples, no_of_features
        m_test, self.n = x_test.shape

        # initialize Y_predict
        y_predict = np.zeros(m_test)

        for i in range(m_test):
            # find the k nearest neighbors from current test example
            neighbors = np.zeros(self.k)
            neighbors = self.find_neighbors(x_test[i])

            # most frequent class in k neighbors
            y_predict[i] = mode(neighbors)[0][0]

        return y_predict

    # function to find the k nearest neighbors to current test example
    def find_neighbors(self, x):
        # calculate all the euclidean distances between current test example x and training set X_train
        euclidean_distances = np.zeros(self.m)
        x = np.reshape(x, (1, self.n))

        # create x for broadcasting with X_train
        x = np.repeat(x, repeats=self.m, axis=0)

        # (x - y)^2 = x^2 + y^2 - 2 * x * y
        euclidean_distances = np.sum((np.square(x) + np.square(self.x_train) - 2 * self.x_train * x), axis=1)

        # sort y_train according to euclidean_distance_array and store into y_train_sorted
        ids = euclidean_distances.argsort()

        return self.y_train[ids[:self.k]]

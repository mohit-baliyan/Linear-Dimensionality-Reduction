import numpy as np


class Fisher:

    def fit(self, X, y):

        # samples segregation based on class
        X_class1 = X[np.where(y == 1)[0], :]
        X_class0 = X[np.where(y == 0)[0], :]

        # calculate mean
        mean_X_class1 = X_class1.mean(axis=0)
        mean_X_class0 = X_class0.mean(axis=0)

        # calculate covariance matrices
        cov_X_class1 = np.cov(X_class1, rowvar=False)
        cov_X_class0 = np.cov(X_class0, rowvar=False)
        matrix = cov_X_class1 + cov_X_class0
        matrix = matrix + 0.001 * max(abs(np.diag(matrix))) * np.eye(np.size(mean_X_class1))

        # calculate fisher directions
        diff = np.array(mean_X_class1 - mean_X_class0)
        self.direction = np.linalg.inv(matrix).dot(diff)

        # normalize fisher directions
        self.direction = self.direction / np.sqrt(np.sum(np.power(self.direction, 2)))

    # hypothetical function  h(x)
    def predict(self, X):

        z = np.dot(X, self.direction)
        return z
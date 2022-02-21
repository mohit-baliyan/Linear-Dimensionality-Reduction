import numpy as np


class Fisher:

    def fit(self, x, y):

        # samples segregation based on class
        x_class1 = x[np.where(y == 1)[0], :]
        x_class0 = x[np.where(y == 0)[0], :]

        # calculate mean
        mean_x_class1 = x_class1.mean(axis=0)
        mean_x_class0 = x_class0.mean(axis=0)

        # calculate covariance matrices
        cov_x_class1 = np.cov(x_class1, rowvar=False)
        cov_x_class0 = np.cov(x_class0, rowvar=False)
        matrix = cov_x_class1 + cov_x_class0
        matrix = matrix + 0.001 * max(abs(np.diag(matrix))) * np.eye(np.size(mean_x_class1))

        # calculate fisher directions
        diff = np.array(mean_x_class1 - mean_x_class0)
        self.direction = np.linalg.inv(matrix).dot(diff)

        # normalize fisher directions
        self.direction = self.direction / np.sqrt(np.sum(np.square(self.direction)))

    # hypothetical function  h(x)
    def predict(self, x):

        z = np.dot(x, self.direction)
        return z
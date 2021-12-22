import numpy as np


# Logistic Regression
class LogitRegression:

    def __init__(self, learning_rate, iterations):

        # initialize hyper parameters
        self.learning_rate = learning_rate
        self.iterations = iterations

    # Function for model training
    def fit(self, X, Y):

        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape

        # weight initialization
        self.W = np.zeros(self.n)
        self.b = 0

        self.X = X
        self.Y = Y

        # gradient descent learning
        for i in range(self.iterations):
            self.update_weights()

        return self

    # Helper function to update weights in gradient descent
    def update_weights(self):

        A = 1 / (1 + np.exp(- (self.X.dot(self.W) + self.b)))

        # calculate gradients
        tmp = (A - self.Y.T)
        tmp = np.reshape(tmp, self.m)
        dW = np.dot(self.X.T, tmp) / self.m
        db = np.sum(tmp) / self.m

        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    # Hypothetical function  h( x )
    def predict(self, X):

        Z = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
        return Z

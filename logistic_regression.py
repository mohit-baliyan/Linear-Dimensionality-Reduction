import numpy as np


# Logistic Regression
class LogitRegression:

    def __init__(self, learning_rate):
        # initialize hyper parameters
        self.learning_rate = learning_rate

    # train model
    def fit(self, X, y):

        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape
        # weight, X, y, cost function initialization
        self.w = np.random.rand(self.n)
        self.b = 0
        self.X = X
        self.y = y
        self.J1 = 0
        self.J2 = 0

        # training
        while self.learning_rate > 0.000001:

            # compute cost function and gradients
            self.J1 = self.cost_function()
            self.gradients()
            self.J1 = self.cost_function()

            # update weights
            self.w = self.w - self.learning_rate * self.dw
            self.b = self.b - self.learning_rate * self.db

            # compute cost function
            self.J2 = self.cost_function()

            # if converge increase the degree of convergence
            if self.J2 < self.J1:
                while self.J2 <= self.J1:
                    # update weights
                    self.w = self.w - self.learning_rate * self.dw
                    self.b = self.b - self.learning_rate * self.db
                    self.learning_rate = self.learning_rate * 2
                    self.J1 = self.J2
                    self.J2 = self.cost_function()
                self.learning_rate = self.learning_rate / 2
                self.w = self.w + self.learning_rate * self.dw
                self.b = self.b + self.learning_rate * self.db

            else:
                while self.J2 > self.J1:
                    self.learning_rate = self.learning_rate / 2
                    self.w = self.w + self.learning_rate * self.dw
                    self.b = self.b + self.learning_rate * self.db
                    self.J2 = self.cost_function()

    def cost_function(self):
        # compute a or y_hat
        self.a = self.predict(self.X)
        # compute cost function
        J = -1 / self.m * np.sum(self.y * np.log(self.a) + (1 - self.y) * (np.log(1 - self.a)))
        return J

    def gradients(self):
        # calculate dJ/dw, dJ/db
        tmp = (self.a - self.y.T)
        tmp = np.reshape(tmp, self.m)
        self.dw = np.dot(self.X.T, tmp) / self.m
        self.db = np.sum(tmp) / self.m
        return self

    # hypothetical function  h(x)
    def predict(self, X):
        z = 1 / (1 + np.exp(- (X.dot(self.w) + self.b)))
        return z

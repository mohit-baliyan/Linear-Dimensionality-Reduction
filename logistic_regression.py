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

        # weight initialization
        self.w = np.random.rand(self.n)
        self.b = 0

        # initiation of accuracy measures
        w_old = 0
        b_old = 0

        # training
        while ((np.sum(np.square(w_old - self.w) + np.sum(np.square(b_old - self.b)))) /
               (np.sum(np.square((w_old + self.w) / 2)) - np.sum(np.square((b_old + self.b) / 2))) > 0.0001):

            # save old weights
            w_old = self.w
            b_old = self.b

            # compute cost function and gradients
            self.J1 = self.cost_function(X, y)
            self.gradients(X, y)

            # update weights
            self.w = self.w - self.learning_rate * self.dw
            self.b = self.b - self.learning_rate * self.db

            # compute cost function
            self.J2 = self.cost_function(X, y)

            # if converged, increase the degree of convergence
            if self.J2 <= self.J1:
                while self.J2 <= self.J1:
                    # update weights
                    self.w = self.w - self.learning_rate * self.dw
                    self.b = self.b - self.learning_rate * self.db
                    self.learning_rate = self.learning_rate * 2
                    self.J1 = self.J2
                    self.J2 = self.cost_function(X, y)
                self.learning_rate = self.learning_rate / 2
                self.w = self.w + self.learning_rate * self.dw
                self.b = self.b + self.learning_rate * self.db

            else:
                while self.J2 > self.J1:
                    self.learning_rate = self.learning_rate / 2
                    self.w = self.w + self.learning_rate * self.dw
                    self.b = self.b + self.learning_rate * self.db
                    self.J2 = self.cost_function(X, y)

    def cost_function(self, X, y):
        # compute a or y_hat
        self.a = self.predict(X)
        # compute cost function
        J = -1 / self.m * np.sum(y * np.log(self.a) + (1 - y) * (np.log(1 - self.a)))
        return J

    def gradients(self, X, y):
        # calculate dJ/dw, dJ/db
        tmp = (self.a - y.T)
        tmp = np.reshape(tmp, self.m)
        self.dw = np.dot(X.T, tmp) / self.m
        self.db = np.sum(tmp) / self.m
        return self

    # hypothetical function  h(x)
    def predict(self, X):
        z = 1 / (1 + np.exp(- (X.dot(self.w) + self.b)))
        return z
import numpy as np


# Logistic Regression
class LogitRegression:

    def __init__(self, learning_rate):

        # initialize hyper parameters
        self.db = None
        self.dw = None
        self.b = None
        self.w = None
        self.n = None
        self.m = None
        self.learning_rate = learning_rate

    # train model
    def fit(self, x, y):

        # no_of_training_examples, no_of_features
        self.m, self.n = x.shape

        # weight initialization
        self.w = np.random.rand(self.n)
        self.b = 0

        # initiation of accuracy measures
        w_old = 0
        b_old = 0

        # training
        while ((np.sum(np.square(w_old - self.w) + np.sum(np.square(b_old - self.b)))) /
               (np.sum(np.square((w_old + self.w) / 2)) + np.sum(np.square((b_old + self.b) / 2))) > 1e-9):

            # save old weights
            w_old = self.w
            b_old = self.b

            # compute cost function and gradients
            j1 = self.cost_function(x, y)
            self.gradients(x, y)

            # update weights
            self.w = self.w - self.learning_rate * self.dw
            self.b = self.b - self.learning_rate * self.db

            # compute cost function
            j2 = self.cost_function(x, y)

            # if converged, increase the degree of convergence
            if j2 <= j1:
                while j2 <= j1:
                    # update weights
                    self.w = self.w - self.learning_rate * self.dw
                    self.b = self.b - self.learning_rate * self.db
                    self.learning_rate = self.learning_rate * 2
                    j1 = j2
                    j2 = self.cost_function(x, y)
                self.learning_rate = self.learning_rate / 2
                self.w = self.w + self.learning_rate * self.dw
                self.b = self.b + self.learning_rate * self.db

            else:
                while j2 > j1:
                    self.learning_rate = self.learning_rate / 2
                    self.w = self.w + self.learning_rate * self.dw
                    self.b = self.b + self.learning_rate * self.db
                    j2 = self.cost_function(x, y)

    def cost_function(self, x, y):
        # compute a or y_hat
        a = self.predict(x)
        # compute cost function
        j = (-1 / self.m) * np.sum(y * np.log(a) + (1 - y) * (np.log(1 - a)))
        return j

    def gradients(self, x, y):
        # calculate dJ/dw, dJ/db
        a = self.predict(x)
        tmp = (a - y.T)
        tmp = np.reshape(tmp, self.m)
        self.dw = np.dot(x.T, tmp) / self.m
        self.db = np.sum(tmp) / self.m
        return self

    # hypothetical function  h(x)
    def predict(self, x):
        z = 1 / (1 + np.exp(- (x.dot(self.w) + self.b)))
        return z

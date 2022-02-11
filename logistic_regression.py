import numpy as np


# Logistic Regression
class LogitRegression:

    def __init__(self, learning_rate):

        # initialize hyper parameters
        self.learning_rate = learning_rate

    # train model
    def fit(self, X, y):

        # local reference to avoid unnecessary reference
        learning_rate = self.learning_rate

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
               (np.sum(np.square((w_old + self.w) / 2)) + np.sum(np.square((b_old + self.b) / 2))) > 1e-9):

            # save old weights
            w_old = self.w
            b_old = self.b

            # compute cost function and gradients
            J1 = self.cost_function(X, y)
            self.gradients(X, y)

            # update weights
            self.w = self.w - learning_rate * self.dw
            self.b = self.b - learning_rate * self.db

            # compute cost function
            J2 = self.cost_function(X, y)

            # if converged, increase the degree of convergence
            if J2 <= J1:
                while J2 <= J1:
                    # update weights
                    self.w = self.w - learning_rate * self.dw
                    self.b = self.b - learning_rate * self.db
                    learning_rate = learning_rate * 2
                    J1 = J2
                    J2 = self.cost_function(X, y)
                learning_rate = learning_rate / 2
                self.w = self.w + learning_rate * self.dw
                self.b = self.b + learning_rate * self.db

            else:
                while J2 > J1:
                    learning_rate = learning_rate / 2
                    self.w = self.w + learning_rate * self.dw
                    self.b = self.b + learning_rate * self.db
                    J2 = self.cost_function(X, y)

    def cost_function(self, X, y):
        # compute a or y_hat
        a = self.predict(X)
        # compute cost function
        J = (-1 / self.m) * np.sum(y * np.log(a) + (1 - y) * (np.log(1 - a)))
        return J

    def gradients(self, X, y):
        # calculate dJ/dw, dJ/db
        a = self.predict(X)
        tmp = (a - y.T)
        tmp = np.reshape(tmp, self.m)
        self.dw = np.dot(X.T, tmp) / self.m
        self.db = np.sum(tmp) / self.m
        return self

    # hypothetical function  h(x)
    def predict(self, X):
        z = 1 / (1 + np.exp(- (X.dot(self.w) + self.b)))
        return z
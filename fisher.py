import numpy as np
import pandas as pd


class Fisher:

    def fit(self, X, y):

        self.X = X
        self.y = y

        # samples segregation based on class
        df = pd.DataFrame(X)
        df['y'] = y
        df_class1 = df[y == 1]
        df_class0 = df[y == 0]
        X_class1 = df_class1.iloc[:, :-1]
        X_class0 = df_class0.iloc[:, :-1]

        # calculate mean
        mean_X_class1 = X_class1.mean(axis=0)
        mean_X_class0 = X_class0.mean(axis=0)

        # calculate covariance matrices
        cov_X_class1 = np.cov(X_class1, rowvar=False)
        cov_X_class0 = np.cov(X_class0, rowvar=False)

        # calculate fisher directions

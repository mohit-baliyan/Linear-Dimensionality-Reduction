import os
import pandas as pd
from knn import KNN
from scipy.io import loadmat
from scipy.stats import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import balanced_accuracy_score


def best_k(file):

    # load database
    data = loadmat('Databases/' + file)
    x = data['X']
    y = data['Y']

    # standardise
    x = stats.zscore(x)

    best_n = 0
    best_score = 0

    # select best K
    for i in range(1, 30, 2):

        # initialize leave one out cross validation object
        cv = LeaveOneOut()

        # enumerate splits
        y_true, y_pred = list(), list()

        for train_ix, test_ix in cv.split(x):

            # split data
            x_train, x_test = x[train_ix, :], x[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]

            # fit model
            model = KNN(k=i)
            model.fit(x_train, y_train)

            # evaluate model
            yhat = model.predict(x_test)

            # store
            y_true.append(y_test[0])
            y_pred.append(yhat[0])

        # calculate accuracy
        score = balanced_accuracy_score(y_true, y_pred)

        # select best score and best K
        if score > best_score:
            best_score = score
            best_n = i

    # delete caches
    del data, x, y, best_score, cv, y_true, y_pred, train_ix, test_ix, x_train, x_test, y_train, y_test, model, \
        yhat, score

    return best_n


def database():

    k = best_k('LfW_faces.mat')
    print('LfW_faces.mat', k)



def main():

    # K selection
    database()


if __name__ == "__main__":

    main()
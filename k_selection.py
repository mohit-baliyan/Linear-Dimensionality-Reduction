import os
from knn import KNN
import pandas as pd
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import balanced_accuracy_score


def best_k(file):
    # load database
    data = loadmat('Databases/' + file)
    x = data['X']
    y = data['Y']

    # standardise
    x = zscore(x)

    best_n = 0
    best_score = 0

    # select best K
    for i in range(1, 30, 2):

        # initialize leave one out cross validation object
        cv = LeaveOneOut()

        # enumerate splits
        y_true, y_predict = list(), list()

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
            y_predict.append(yhat[0])

        # calculate accuracy
        score = balanced_accuracy_score(y_true, y_predict)

        # select best score and best K
        if score > best_score:
            best_score = score
            best_n = i

    return best_n


def main():
    # create list of databases
    files = os.listdir('./Databases/')

    # initialize lists to store each database name and best k for that database
    databases = []
    K = []

    for file in files:
        # find best K
        databases.append(file)
        K.append(best_k(file))
        print(file, "success")

    # create dictionary, then convert into pandas dataframe to further save as a csv file
    dictionary = {'Databases': databases, 'K': K}

    df = pd.DataFrame(dictionary)
    df.to_csv('best_K.csv')

    print("succeed")


if __name__ == "__main__":
    main()

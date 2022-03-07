import warnings
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import zscore
from logistic_regression import LogitRegression
from sklearn.linear_model import LogisticRegression
from threshold_selection import threshold_selection
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")


def balanced_accuracy(file):
    # load database
    data = loadmat('Databases/' + file)
    x = data['X']
    y = data['Y']

    # standardise
    x = zscore(x)

    total_balanced_accuracy = 0

    # stratified 10-fold cross validation
    for j in range(1, 11):
        # selecting indices of training set
        train_set = pd.read_csv('Folds-Databases/' + file + '/train_fold_' + str(j) + '.txt', header=None)
        train_index = train_set.to_numpy()
        mt, nf = train_index.shape
        train_index = np.reshape(train_index, mt)
        train = list(train_index)
        train = [int(item) for item in train]

        # selecting indices of test set for jth fold
        test_set = pd.read_csv('Folds-Databases/' + file + '/test_fold_' + str(j) + '.txt', header=None)
        test_index = test_set.to_numpy()
        mv, nf = test_index.shape
        test_index = np.reshape(test_index, mv)
        test = list(test_index)
        test = [int(item) for item in test]

        # getting x, y for current folds from indices above
        x_train = np.take(x, train, axis=0)
        x_test = np.take(x, test, axis=0)

        y_train = np.take(y, train, axis=0)
        y_test = np.take(y, test, axis=0)

        # without dimensionality reduction
        model1 = LogitRegression(0.01)
        model1.fit(x_train, y_train)
        # predict on training set
        y_train_predict = model1.predict(x_train)
        # select optimal threshold on training dataset
        threshold = threshold_selection(y_train_predict, y_train)
        # predict on test set
        y_test_predict = model1.predict(x_test)
        y_test_predict = np.where(y_test_predict > threshold, 1, 0)
        total_balanced_accuracy = total_balanced_accuracy + balanced_accuracy_score(y_test, y_test_predict)

    return total_balanced_accuracy / 10


def sklearn_accuracy(file):
    # load database
    data = loadmat('Databases/' + file)
    x = data['X']
    y = data['Y']

    # standardise
    x = zscore(x)

    total_balanced_accuracy = 0

    # stratified 10-fold cross validation
    for j in range(1, 11):
        # selecting indices of training set
        train_set = pd.read_csv('Folds-Databases/' + file + '/train_fold_' + str(j) + '.txt', header=None)
        train_index = train_set.to_numpy()
        mt, nf = train_index.shape
        train_index = np.reshape(train_index, mt)
        train = list(train_index)
        train = [int(item) for item in train]

        # selecting indices of test set for jth fold
        test_set = pd.read_csv('Folds-Databases/' + file + '/test_fold_' + str(j) + '.txt', header=None)
        test_index = test_set.to_numpy()
        mv, nf = test_index.shape
        test_index = np.reshape(test_index, mv)
        test = list(test_index)
        test = [int(item) for item in test]

        # getting x, y for current folds from indices above
        x_train = np.take(x, train, axis=0)
        x_test = np.take(x, test, axis=0)

        y_train = np.take(y, train, axis=0)
        y_test = np.take(y, test, axis=0)

        # without dimensionality reduction
        model1 = LogisticRegression()
        model1.fit(x_train, y_train)
        # predict on test set
        y_test_predict = model1.predict(x_test)
        total_balanced_accuracy = total_balanced_accuracy + balanced_accuracy_score(y_test, y_test_predict)

    return total_balanced_accuracy / 10


def main():

    print(balanced_accuracy('Banknote.mat'))

    print(sklearn_accuracy('EggEyeState.mat'))


if __name__ == "__main__":
    main()

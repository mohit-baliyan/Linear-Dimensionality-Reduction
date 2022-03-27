import numpy as np
from pca import PCA
import pandas as pd
from scipy.io import loadmat
from scipy.stats import zscore
from logistic_regression import LogitRegression
from sklearn.metrics import balanced_accuracy_score
from threshold_selection import threshold_selection
from sklearn.linear_model import LogisticRegression


def balanced_accuracy(file, dimensions):
    # load database
    data = loadmat('Databases/' + file)
    x = data['X']
    y = data['Y']

    # standardise
    x = zscore(x)

    # transformation
    pca = PCA()
    dim = dimensions.loc[dimensions['Databases'] == file]
    x_kaiser = pca.transformation(x, dim["PCA-K"].values[0])
    x_bs = pca.transformation(x, dim["PCA-BS"].values[0])
    x_cn = pca.transformation(x, dim["PCA-CN"].values[0])

    # n repetitions on stratified K fold cross validation
    n = 10

    total_balanced_accuracy = 0
    total_balanced_accuracy_k = 0
    total_balanced_accuracy_bs = 0
    total_balanced_accuracy_cn = 0

    for i in range(0, n):

        balance_accuracy = 0
        balanced_accuracy_k = 0
        balanced_accuracy_bs = 0
        balanced_accuracy_cn = 0

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

            x_train_k = np.take(x_kaiser, train, axis=0)
            x_test_k = np.take(x_kaiser, test, axis=0)

            x_train_bs = np.take(x_bs, train, axis=0)
            x_test_bs = np.take(x_bs, test, axis=0)

            x_train_cn = np.take(x_cn, train, axis=0)
            x_test_cn = np.take(x_cn, test, axis=0)

            y_train = np.take(y, train, axis=0)
            y_test = np.take(y, test, axis=0)

            model = LogitRegression(0.01, 1e-19)
            # without dimensionality reduction
            sk1 = LogisticRegression()
            sk1.fit(x_train, y_train)
            w1, b1 = sk1.coef_, sk1.intercept_
            # predict on training set
            y_train_predict = model.predict_sklearn(x_train, w1, b1)
            # select optimal threshold on training dataset
            threshold = threshold_selection(y_train_predict, y_train)
            # predict on test set
            y_test_predict = model.predict_sklearn(x_test, w1, b1)
            y_test_predict = np.where(y_test_predict > threshold, 1, 0)
            balance_accuracy = balance_accuracy + balanced_accuracy_score(y_test, y_test_predict)

            # for kaiser
            sk2 = LogisticRegression()
            sk2.fit(x_train_k, y_train)
            w2, b2 = sk2.coef_, sk2.intercept_
            # predict on training set
            y_train_predict_k = model.predict_sklearn(x_train_k, w2, b2)
            # select optimal threshold on training dataset
            threshold_k = threshold_selection(y_train_predict_k, y_train)
            # predict on test set
            y_test_predict_k = model.predict_sklearn(x_test_k, w2, b2)
            y_test_predict_k = np.where(y_test_predict_k > threshold_k, 1, 0)
            balanced_accuracy_k = balanced_accuracy_k + balanced_accuracy_score(y_test, y_test_predict_k)

            # for BS
            sk3 = LogisticRegression()
            sk3.fit(x_train_bs, y_train)
            w3, b3 = sk3.coef_, sk3.intercept_
            # predict on training set
            y_train_predict_bs = model.predict_sklearn(x_train_bs, w3, b3)
            # select optimal threshold on training dataset
            threshold_bs = threshold_selection(y_train_predict_bs, y_train)
            y_test_predict_bs = model.predict_sklearn(x_test_bs, w3, b3)
            y_test_predict_bs = np.where(y_test_predict_bs > threshold_bs, 1, 0)
            balanced_accuracy_bs = balanced_accuracy_bs + balanced_accuracy_score(y_test, y_test_predict_bs)

            # for CN
            sk4 = LogisticRegression()
            sk4.fit(x_train_cn, y_train)
            w4, b4 = sk4.coef_, sk4.intercept_
            # predict on training set
            y_train_predict_cn = model.predict_sklearn(x_train_cn, w4, b4)
            # select optimal threshold on training dataset
            threshold_cn = threshold_selection(y_train_predict_cn, y_train)
            y_test_predict_cn = model.predict_sklearn(x_test_cn, w4, b4)
            y_test_predict_cn = np.where(y_test_predict_cn > threshold_cn, 1, 0)
            balanced_accuracy_cn = balanced_accuracy_cn + balanced_accuracy_score(y_test, y_test_predict_cn)

        total_balanced_accuracy = total_balanced_accuracy + balance_accuracy / 10
        total_balanced_accuracy_k = total_balanced_accuracy_k + balanced_accuracy_k / 10
        total_balanced_accuracy_bs = total_balanced_accuracy_bs + balanced_accuracy_bs / 10
        total_balanced_accuracy_cn = total_balanced_accuracy_cn + balanced_accuracy_cn / 10

    # delete cache
    del data, x, y, pca, dim, x_kaiser, x_bs, x_cn, n, i, balance_accuracy, balanced_accuracy_k, balanced_accuracy_bs, \
        balanced_accuracy_cn, j, train_set, train_index, mt, nf, train, test_set, test_index, mv, test, x_train, \
        x_test, y_train, y_test, x_train_k, x_test_k, x_train_bs, x_test_bs, x_train_cn, x_test_cn, model, \
        y_train_predict, y_test_predict, y_train_predict_k, y_test_predict_k, y_train_predict_bs, \
        y_test_predict_bs, y_train_predict_cn, y_test_predict_cn, sk1, sk2, sk3, sk4

    return (total_balanced_accuracy / 10,
            total_balanced_accuracy_k / 10,
            total_balanced_accuracy_bs / 10,
            total_balanced_accuracy_cn / 10)


def main():
    # read dimensions.csv to read number of components
    dimensions = pd.read_csv('dimensions.csv')

    b, b_k, b_bs, b_cn = balanced_accuracy('Musk.mat', dimensions)

    print("b : ", round(b, 4))
    print("b_k : ", round(b_k, 4))
    print("b_bs : ", round(b_bs, 4))
    print("b_cn : ", round(b_cn, 4))


if __name__ == "__main__":
    main()

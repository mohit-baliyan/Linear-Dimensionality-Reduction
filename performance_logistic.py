import os
import numpy as np
from pca import PCA
import pandas as pd
from scipy.io import loadmat
from scipy.stats import stats
from logistic_regression import LogitRegression
from sklearn.metrics import balanced_accuracy_score
from threshold_selection import threshold_selection


def balanced_accuracy(file, dimensions):
    # load database
    data = loadmat('Databases/' + file)
    X = data['X']
    y = data['Y']

    # standardise
    X = stats.zscore(X)

    # transformation
    pca = PCA()
    dim = dimensions.loc[dimensions['Databases'] == file]
    X_Kaiser = pca.transformation(X, dim["PCA-K"].values[0])
    X_BS = pca.transformation(X, dim["PCA-BS"].values[0])
    X_CN = pca.transformation(X, dim["PCA-CN"].values[0])

    # save in pandas dataframe
    df = pd.DataFrame.from_records(X)
    df['y'] = y

    df_K = pd.DataFrame.from_records(X_Kaiser)
    df_BS = pd.DataFrame.from_records(X_BS)
    df_CN = pd.DataFrame.from_records(X_CN)

    # n repetitions on stratified K fold cross validation
    n = 10

    total_balanced_accuracy = 0
    total_balanced_accuracy_K = 0
    total_balanced_accuracy_BS = 0
    total_balanced_accuracy_CN = 0

    for i in range(0, n):

        print(i, "iteration initiated")

        balanced_accuracy = 0
        balanced_accuracy_K = 0
        balanced_accuracy_BS = 0
        balanced_accuracy_CN = 0

        # stratified 10-fold cross validation
        for j in range(1, 11):

            print(j, "fold initiated")

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

            # getting X, y for current folds from indices above
            df_train = df[df.index.isin(train)]
            df_test = df[df.index.isin(test)]

            df_train_K = df_K[df_K.index.isin(train)]
            df_test_K = df_K[df_K.index.isin(test)]

            df_train_BS = df_BS[df_BS.index.isin(train)]
            df_test_BS = df_BS[df_BS.index.isin(test)]

            df_train_CN = df_CN[df_CN.index.isin(train)]
            df_test_CN = df_CN[df_CN.index.isin(test)]

            X_train = df_train.iloc[:, :-1]
            X_test = df_test.iloc[:, :-1]
            y_train = df_train['y']
            y_test = df_test['y']

            X_train_K = df_train_K.iloc[:, :-1]
            X_test_K = df_test_K.iloc[:, :-1]

            X_train_BS = df_train_BS.iloc[:, :-1]
            X_test_BS = df_test_BS.iloc[:, :-1]

            X_train_CN = df_train_CN.iloc[:, :-1]
            X_test_CN = df_test_CN.iloc[:, :-1]

            # transform X_train, X_test
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            X_train_K = X_train_K.to_numpy()
            X_test_K = X_test_K.to_numpy()

            X_train_BS = X_train_BS.to_numpy()
            X_test_BS = X_test_BS.to_numpy()

            X_train_CN = X_train_CN.to_numpy()
            X_test_CN = X_test_CN.to_numpy()

            # without dimensionality reduction
            model1 = LogitRegression(learning_rate=0.01)
            model1.fit(X_train, y_train)
            # predict on training set
            y_train_predict = model1.predict(X_train)
            # select optimal threshold on training dataset
            threshold = threshold_selection(y_train_predict, y_train)
            # predict on test set
            y_test_predict = model1.predict(X_test)
            y_test_predict = np.where(y_test_predict > threshold, 1, 0)
            balanced_accuracy = balanced_accuracy + balanced_accuracy_score(y_test, y_test_predict)

            # for kaiser
            model2 = LogitRegression(learning_rate=0.01)
            model2.fit(X_train_K, y_train)
            # predict on training set
            y_train_predict_K = model2.predict(X_train_K)
            # select optimal threshold on training dataset
            threshold_K = threshold_selection(y_train_predict_K, y_train)
            # predict on test set
            y_test_predict_K = model2.predict(X_test_K)
            y_test_predict_K = np.where(y_test_predict_K > threshold_K, 1, 0)
            balanced_accuracy_K = balanced_accuracy_K + balanced_accuracy_score(y_test, y_test_predict_K)

            # for BS
            model3 = LogitRegression(learning_rate=0.01)
            model3.fit(X_train_BS, y_train)
            # predict on training set
            y_train_predict_BS = model3.predict(X_train_BS)
            # select optimal threshold on training dataset
            threshold_BS = threshold_selection(y_train_predict_BS, y_train)
            y_test_predict_BS = model3.predict(X_test_BS)
            y_test_predict_BS = np.where(y_test_predict_BS > threshold_BS, 1, 0)
            balanced_accuracy_BS = balanced_accuracy_BS + balanced_accuracy_score(y_test, y_test_predict_BS)

            # for CN
            model4 = LogitRegression(learning_rate=0.01)
            model4.fit(X_train_CN, y_train)
            # predict on training set
            y_train_predict_CN = model4.predict(X_train_CN)
            # select optimal threshold on training dataset
            threshold_CN = threshold_selection(y_train_predict_CN, y_train)
            y_test_predict_CN = model4.predict(X_test_CN)
            y_test_predict_CN = np.where(y_test_predict_CN > threshold_CN, 1, 0)
            balanced_accuracy_CN = balanced_accuracy_CN + balanced_accuracy_score(y_test, y_test_predict_CN)

            print(j, "folds executed ")

        print(i, "iteration executed")

        total_balanced_accuracy = total_balanced_accuracy + balanced_accuracy / 10
        total_balanced_accuracy_K = total_balanced_accuracy_K + balanced_accuracy_K / 10
        total_balanced_accuracy_BS = total_balanced_accuracy_BS + balanced_accuracy_BS / 10
        total_balanced_accuracy_CN = total_balanced_accuracy_CN + balanced_accuracy_CN / 10

    # delete cache
    del data, X, y, pca, dim, X_Kaiser, X_BS, X_CN, df, df_K, df_BS, df_CN, n, balanced_accuracy, balanced_accuracy_K, \
        balanced_accuracy_BS, balanced_accuracy_CN, j, train_set, train_index, mt, nf, train, test_set, test_index, \
        mv, test, df_train, df_test, df_train_K, df_test_K, df_train_BS, df_test_BS, df_train_CN, df_test_CN, X_train, \
        X_test, y_train, y_test, X_train_K, X_test_K, X_train_BS, X_test_BS, X_train_CN, X_test_CN, model1, \
        y_train_predict, y_test_predict, model2, y_train_predict_K, y_test_predict_K, model3, y_train_predict_BS, \
        y_test_predict_BS, model4, y_train_predict_CN, y_test_predict_CN

    return (total_balanced_accuracy / 10,
            total_balanced_accuracy_K / 10,
            total_balanced_accuracy_BS / 10,
            total_balanced_accuracy_CN / 10)


def logistic_performance():
    # read files of folded databases
    files = os.listdir('./Folds-Databases/')

    # read dimensions.csv to read number of components
    dimensions = pd.read_csv('dimensions.csv')

    # initialize lists to store results
    Databases = []
    BA = []
    BA_K = []
    BA_BS = []
    BA_CN = []

    for file in files:

        print(file, "initiated dataset")

        Databases.append(file)
        ba, ba_k, ba_cn, ba_bs = balanced_accuracy(file, dimensions)

        print(ba, ba_k, ba_bs,  ba_cn)

        BA.append(ba)
        BA_K.append(ba_k)
        BA_BS.append(ba_bs)
        BA_CN.append(ba_cn)

        print(file, "dataset executed")

    # create dictionary, then convert into pandas dataframe to further save as a csv file
    dictionary = {

        'Databases': Databases,
        'BA': BA,
        'BA_K': BA_K,
        'BA_BS': BA_BS,
        'BA_CN': BA_CN

    }

    df = pd.DataFrame(dictionary)
    df.to_csv('logistic_results.csv')


def main():
    # results from logistic
    logistic_performance()


if __name__ == "__main__":
    main()

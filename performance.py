import os
import numpy as np
from pca import PCA
import pandas as pd
from scipy import stats
from fisher import Fisher
from scipy.io import loadmat
from logistic_regression import LogitRegression
from sklearn.metrics import balanced_accuracy_score
from threshold_selection import threshold_selection


def performance_logistic(file, dimensions):
    # load database
    data = loadmat('Databases/' + file)
    X = data['X']
    y = data['Y']
    # standardise
    X = stats.zscore(X)

    # save in pandas dataframe
    df = pd.DataFrame.from_records(X)
    df['y'] = y

    # initialize PCA
    pca = PCA()

    # n repetitions on stratified K fold cross validation
    n = 10
    total_balanced_accuracy_k = 0
    total_balanced_accuracy_bs = 0
    total_balanced_accuracy_cn = 0

    for i in range(0, n):

        balanced_accuracy_k = 0
        balanced_accuracy_bs = 0
        balanced_accuracy_cn = 0

        # stratified 10-fold cross validation
        for j in range(1, 11):
            # selecting indices of training set
            train_set = pd.read_csv('Folds-Databases/' + file + '/train_fold_' + str(j) + '.txt', header=None)
            train_index = train_set.to_numpy()
            mt, n = train_index.shape
            train_index = np.reshape(train_index, mt)
            train = list(train_index)
            train = [int(item) for item in train]

            # selecting indices of test set for jth fold
            test_set = pd.read_csv('Folds-Databases/' + file + '/test_fold_' + str(j) + '.txt', header=None)
            test_index = test_set.to_numpy()
            mv, n = test_index.shape
            test_index = np.reshape(test_index, mv)
            test = list(test_index)
            test = [int(item) for item in test]

            # getting X, y for current folds from indices above
            df_train = df[df.index.isin(train)]
            df_test = df[df.index.isin(test)]

            X_train = df_train.iloc[:, :-1]
            X_test = df_test.iloc[:, :-1]
            y_train = df_train['y']
            y_test = df_test['y']

            # transform X_train, X_test
            dim = dimensions.loc[dimensions['Databases'] == file]
            X_kaiser_train = pca.transformation(X_train, dim["PCA-K"].values[0])
            X_kaiser_test = pca.transformation(X_test, dim["PCA-K"].values[0])
            X_bs_train = pca.transformation(X_train, dim["PCA-BS"].values[0])
            X_bs_test = pca.transformation(X_test, dim["PCA-BS"].values[0])
            X_cn_train = pca.transformation(X_train, dim["PCA-CN"].values[0])
            X_cn_test = pca.transformation(X_test, dim["PCA-CN"].values[0])

            # model training, selecting threshold and prediction

            # for kaiser
            model_kaiser = LogitRegression(learning_rate=0.01)
            model_kaiser.fit(X_kaiser_train, y_train)
            # predict on training set
            y_kaiser = model_kaiser.predict(X_kaiser_train)
            # select optimal threshold on training dataset
            threshold_kaiser = threshold_selection(y_kaiser, y_train)
            y_predict_kaiser = model_kaiser.predict(X_kaiser_test)
            y_predict_kaiser = np.where(y_predict_kaiser > threshold_kaiser, 1, 0)
            balanced_accuracy_k = balanced_accuracy_k + balanced_accuracy_score(y_test, y_predict_kaiser)

            # for broken stick
            model_bs = LogitRegression(learning_rate=0.01)
            model_bs.fit(X_bs_train, y_train)
            # predict on training set
            y_bs = model_bs.predict(X_bs_train)
            # select optimal threshold on training dataset
            threshold_bs = threshold_selection(y_bs, y_train)
            y_predict_bs = model_bs.predict(X_bs_test)
            y_predict_bs = np.where(y_predict_bs > threshold_bs, 1, 0)
            balanced_accuracy_bs = balanced_accuracy_bs + balanced_accuracy_score(y_test, y_predict_bs)

            # for conditional number
            model_cn = LogitRegression(learning_rate=0.01)
            model_cn.fit(X_cn_train, y_train)
            # predict on training set
            y_cn = model_cn.predict(X_cn_train)
            # select optimal threshold on training dataset
            threshold_cn = threshold_selection(y_cn, y_train)
            y_predict_cn = model_cn.predict(X_cn_test)
            y_predict_cn = np.where(y_predict_cn > threshold_cn, 1, 0)
            balanced_accuracy_cn = balanced_accuracy_cn + balanced_accuracy_score(y_test, y_predict_cn)

        total_balanced_accuracy_k = total_balanced_accuracy_k + balanced_accuracy_k / 10
        total_balanced_accuracy_bs = total_balanced_accuracy_bs + balanced_accuracy_bs / 10
        total_balanced_accuracy_cn = total_balanced_accuracy_cn + balanced_accuracy_cn / 10

    return total_balanced_accuracy_k / n, total_balanced_accuracy_bs / n, total_balanced_accuracy_cn / n


def performance_logistic_database():
    # read files of folded databases
    # files = os.listdir('./Folds-Databases/')
    files = ['spect.mat']

    # read dimensions.csv to read number of components
    dimensions = pd.read_csv('dimensions.csv')

    # initialize lists to store results
    Databases = []
    BA_K = []
    BA_BS = []
    BA_CN = []

    for file in files:
        Databases.append(file)
        ba_k, ba_cn, ba_bs = performance_logistic(file, dimensions)
        BA_K.append(ba_k)
        BA_BS.append(ba_bs)
        BA_CN.append(ba_cn)

    # create dictionary, then convert into pandas dataframe to further save as a csv file
    dictionary = {

        'Databases': Databases,
        'BA_K': BA_K,
        'BA_BS': BA_BS,
        'BA_CN': BA_CN

    }

    df = pd.DataFrame(dictionary)
    df.to_csv('accuracy_logistic_regression.csv')


def performance_fisher(file, dimensions):
    # load database
    data = loadmat('Databases/' + file)
    X = data['X']
    y = data['Y']
    # standardise
    X = stats.zscore(X)

    # save in pandas dataframe
    df = pd.DataFrame.from_records(X)
    df['y'] = y

    # initialize PCA
    pca = PCA()

    # n repetitions on stratified K fold cross validation
    n = 10
    total_balanced_accuracy_k = 0
    total_balanced_accuracy_bs = 0
    total_balanced_accuracy_cn = 0

    for i in range(0, n):

        balanced_accuracy_k = 0
        balanced_accuracy_bs = 0
        balanced_accuracy_cn = 0

        # stratified 10-fold cross validation
        for j in range(1, 11):
            # selecting indices of training set
            train_set = pd.read_csv('Folds-Databases/' + file + '/train_fold_' + str(j) + '.txt', header=None)
            train_index = train_set.to_numpy()
            mt, n = train_index.shape
            train_index = np.reshape(train_index, mt)
            train = list(train_index)
            train = [int(item) for item in train]

            # selecting indices of test set for jth fold
            test_set = pd.read_csv('Folds-Databases/' + file + '/test_fold_' + str(j) + '.txt', header=None)
            test_index = test_set.to_numpy()
            mv, n = test_index.shape
            test_index = np.reshape(test_index, mv)
            test = list(test_index)
            test = [int(item) for item in test]

            # getting X, y for current folds from indices above
            df_train = df[df.index.isin(train)]
            df_test = df[df.index.isin(test)]

            X_train = df_train.iloc[:, :-1]
            X_test = df_test.iloc[:, :-1]
            y_train = df_train['y']
            y_test = df_test['y']

            # transform X_train, X_test
            dim = dimensions.loc[dimensions['Databases'] == file]
            X_kaiser_train = pca.transformation(X_train, dim["PCA-K"].values[0])
            X_kaiser_test = pca.transformation(X_test, dim["PCA-K"].values[0])
            X_bs_train = pca.transformation(X_train, dim["PCA-BS"].values[0])
            X_bs_test = pca.transformation(X_test, dim["PCA-BS"].values[0])
            X_cn_train = pca.transformation(X_train, dim["PCA-CN"].values[0])
            X_cn_test = pca.transformation(X_test, dim["PCA-CN"].values[0])

            # model training, selecting threshold and prediction

            # for kaiser
            model_kaiser = Fisher()
            model_kaiser.fit(X_kaiser_train, y_train)
            # predict on training set
            y_kaiser = model_kaiser.predict(X_kaiser_train)
            # select optimal threshold on training dataset
            threshold_kaiser = threshold_selection(y_kaiser, y_train)
            y_predict_kaiser = model_kaiser.predict(X_kaiser_test)
            y_predict_kaiser = np.where(y_predict_kaiser > threshold_kaiser, 1, 0)
            balanced_accuracy_k = balanced_accuracy_k + balanced_accuracy_score(y_test, y_predict_kaiser)

            # for broken stick
            model_bs = Fisher()
            model_bs.fit(X_bs_train, y_train)
            # predict on training set
            y_bs = model_bs.predict(X_bs_train)
            # select optimal threshold on training dataset
            threshold_bs = threshold_selection(y_bs, y_train)
            y_predict_bs = model_bs.predict(X_bs_test)
            y_predict_bs = np.where(y_predict_bs > threshold_bs, 1, 0)
            balanced_accuracy_bs = balanced_accuracy_bs + balanced_accuracy_score(y_test, y_predict_bs)

            # for conditional number
            model_cn = Fisher()
            model_cn.fit(X_cn_train, y_train)
            # predict on training set
            y_cn = model_cn.predict(X_cn_train)
            # select optimal threshold on training dataset
            threshold_cn = threshold_selection(y_cn, y_train)
            y_predict_cn = model_cn.predict(X_cn_test)
            y_predict_cn = np.where(y_predict_cn > threshold_cn, 1, 0)
            balanced_accuracy_cn = balanced_accuracy_cn + balanced_accuracy_score(y_test, y_predict_cn)

        total_balanced_accuracy_k = total_balanced_accuracy_k + balanced_accuracy_k / 10
        total_balanced_accuracy_bs = total_balanced_accuracy_bs + balanced_accuracy_bs / 10
        total_balanced_accuracy_cn = total_balanced_accuracy_cn + balanced_accuracy_cn / 10

    return total_balanced_accuracy_k / 10, total_balanced_accuracy_bs / 10, total_balanced_accuracy_cn / 10


def performance_fisher_database():
    # read files of folded databases
    # files = os.listdir('./Folds-Databases/')
    files = ['Immunotherapy.mat']

    # read dimensions.csv to read number of components
    dimensions = pd.read_csv('dimensions.csv')

    # initialize lists to store results
    Databases = []
    BA_K = []
    BA_BS = []
    BA_CN = []

    for file in files:
        Databases.append(file)
        ba_k, ba_cn, ba_bs = performance_fisher(file, dimensions)
        BA_K.append(ba_k)
        BA_BS.append(ba_bs)
        BA_CN.append(ba_cn)

    # create dictionary, then convert into pandas dataframe to further save as a csv file
    dictionary = {

        'Databases': Databases,
        'BA_K': BA_K,
        'BA_BS': BA_BS,
        'BA_CN': BA_CN

    }

    df = pd.DataFrame(dictionary)
    df.to_csv('accuracy_fisher.csv')


def main():
    # results from logistic regression
    # performance_logistic_database()
    performance_fisher_database()


if __name__ == "__main__":
    main()

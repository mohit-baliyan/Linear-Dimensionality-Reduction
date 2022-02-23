# libraries
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
    x = data['X']
    y = data['Y']

    # standardise
    x = stats.zscore(x)

    # transformation
    pca = PCA()
    dim = dimensions.loc[dimensions['Databases'] == file]
    x_kaiser = pca.transformation(x, dim["PCA-K"].values[0])
    x_bs = pca.transformation(x, dim["PCA-BS"].values[0])
    x_cn = pca.transformation(x, dim["PCA-CN"].values[0])

    # save in pandas dataframe
    df = pd.DataFrame.from_records(x)
    df['y'] = y

    df_k = pd.DataFrame.from_records(x_kaiser)
    df_bs = pd.DataFrame.from_records(x_bs)
    df_cn = pd.DataFrame.from_records(x_cn)

    # n repetitions on stratified K fold cross validation
    n = 10

    total_balanced_accuracy = 0
    total_balanced_accuracy_k = 0
    total_balanced_accuracy_bs = 0
    total_balanced_accuracy_cn = 0

    for i in range(0, n):

        balanced_accuracy = 0
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

            # getting X, y for current folds from indices above
            df_train = df[df.index.isin(train)]
            df_test = df[df.index.isin(test)]

            df_train_k = df_k[df_k.index.isin(train)]
            df_test_k = df_k[df_k.index.isin(test)]

            df_train_bs = df_bs[df_bs.index.isin(train)]
            df_test_bs = df_bs[df_bs.index.isin(test)]

            df_train_cn = df_cn[df_cn.index.isin(train)]
            df_test_cn = df_cn[df_cn.index.isin(test)]

            x_train = df_train.iloc[:, :-1]
            x_test = df_test.iloc[:, :-1]
            y_train = df_train['y']
            y_test = df_test['y']

            x_train_k = df_train_k.iloc[:, :]
            x_test_k = df_test_k.iloc[:, :]

            x_train_bs = df_train_bs.iloc[:, :]
            x_test_bs = df_test_bs.iloc[:, :]

            x_train_cn = df_train_cn.iloc[:, :]
            x_test_cn = df_test_cn.iloc[:, :]

            # transform X_train, X_test
            x_train = x_train.to_numpy()
            x_test = x_test.to_numpy()

            x_train_k = x_train_k.to_numpy()
            x_test_k = x_test_k.to_numpy()

            x_train_bs = x_train_bs.to_numpy()
            x_test_bs = x_test_bs.to_numpy()

            x_train_cn = x_train_cn.to_numpy()
            x_test_cn = x_test_cn.to_numpy()

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
            balanced_accuracy = balanced_accuracy + balanced_accuracy_score(y_test, y_test_predict)

            # for kaiser
            model2 = LogitRegression(0.01)
            model2.fit(x_train_k, y_train)
            # predict on training set
            y_train_predict_k = model2.predict(x_train_k)
            # select optimal threshold on training dataset
            threshold_k = threshold_selection(y_train_predict_k, y_train)
            # predict on test set
            y_test_predict_k = model2.predict(x_test_k)
            y_test_predict_k = np.where(y_test_predict_k > threshold_k, 1, 0)
            balanced_accuracy_k = balanced_accuracy_k + balanced_accuracy_score(y_test, y_test_predict_k)

            # for BS
            model3 = LogitRegression(0.01)
            model3.fit(x_train_bs, y_train)
            # predict on training set
            y_train_predict_bs = model3.predict(x_train_bs)
            # select optimal threshold on training dataset
            threshold_bs = threshold_selection(y_train_predict_bs, y_train)
            y_test_predict_bs = model3.predict(x_test_bs)
            y_test_predict_bs = np.where(y_test_predict_bs > threshold_bs, 1, 0)
            balanced_accuracy_bs = balanced_accuracy_bs + balanced_accuracy_score(y_test, y_test_predict_bs)

            # for CN
            model4 = LogitRegression(0.01)
            model4.fit(x_train_cn, y_train)
            # predict on training set
            y_train_predict_cn = model4.predict(x_train_cn)
            # select optimal threshold on training dataset
            threshold_cn = threshold_selection(y_train_predict_cn, y_train)
            y_test_predict_cn = model4.predict(x_test_cn)
            y_test_predict_cn = np.where(y_test_predict_cn > threshold_cn, 1, 0)
            balanced_accuracy_cn = balanced_accuracy_cn + balanced_accuracy_score(y_test, y_test_predict_cn)

        total_balanced_accuracy = total_balanced_accuracy + balanced_accuracy / 10
        total_balanced_accuracy_k = total_balanced_accuracy_k + balanced_accuracy_k / 10
        total_balanced_accuracy_bs = total_balanced_accuracy_bs + balanced_accuracy_bs / 10
        total_balanced_accuracy_cn = total_balanced_accuracy_cn + balanced_accuracy_cn / 10

    # delete cache
    del data, x, y, pca, dim, x_kaiser, x_bs, x_cn, df, df_k, df_bs, df_cn, n, i, balanced_accuracy, \
        balanced_accuracy_k, balanced_accuracy_bs, balanced_accuracy_cn, j, train_set, train_index, mt, nf, train, \
        test_set, test_index, mv, test, df_train, df_test, df_train_k, df_test_k, df_train_bs, df_test_bs, df_train_cn,\
        df_test_cn, x_train, x_test, y_train, y_test, x_train_k, x_test_k, x_train_bs, x_test_bs, x_train_cn, \
        x_test_cn, model1, y_train_predict, y_test_predict, model2, y_train_predict_k, y_test_predict_k, model3, \
        y_train_predict_bs, y_test_predict_bs, model4, y_train_predict_cn, y_test_predict_cn

    return (total_balanced_accuracy / 10,
            total_balanced_accuracy_k / 10,
            total_balanced_accuracy_bs / 10,
            total_balanced_accuracy_cn / 10)


def logistic_performance():

    # read dimensions.csv to read number of components
    dimensions = pd.read_csv('dimensions.csv')

    b, b_k, b_bs, b_cn = balanced_accuracy('blood.mat', dimensions)

    print("b : ", round(b, 4))
    print("b_k : ", round(b_k, 4))
    print("b_bs : ", round(b_bs, 4))
    print("b_cn : ", round(b_cn, 4))


def main():

    # results from logistic
    logistic_performance()


if __name__ == "__main__":
    main()
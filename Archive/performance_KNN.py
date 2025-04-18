import numpy as np
from pca import PCA
import pandas as pd
from knn import KNN
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.metrics import balanced_accuracy_score


def balanced_accuracy(file, dimensions, k):
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

    b_full = []
    b_kaiser = []
    b_bs = []
    b_cn = []

    counter = 0

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

            # without dimensionality reduction
            model1 = KNN(k=k)
            model1.fit(x_train, y_train)
            # predict on test set
            y_test_predict = model1.predict(x_test)
            acc1 = balanced_accuracy_score(y_test, y_test_predict)
            balance_accuracy = balance_accuracy + acc1
            b_full.append(acc1)

            # for kaiser
            model2 = KNN(k=k)
            model2.fit(x_train_k, y_train)
            # predict on test set
            y_test_predict_k = model2.predict(x_test_k)
            acc2 = balanced_accuracy_score(y_test, y_test_predict_k)
            balanced_accuracy_k = balanced_accuracy_k + acc2
            b_kaiser.append(acc2)

            # for BS
            model3 = KNN(k=k)
            model3.fit(x_train_bs, y_train)
            y_test_predict_bs = model3.predict(x_test_bs)
            acc3 = balanced_accuracy_score(y_test, y_test_predict_bs)
            balanced_accuracy_bs = balanced_accuracy_bs + acc3
            b_bs.append(acc3)

            # for CN
            model4 = KNN(k=k)
            model4.fit(x_train_cn, y_train)
            y_test_predict_cn = model4.predict(x_test_cn)
            acc4 = balanced_accuracy_score(y_test, y_test_predict_cn)
            balanced_accuracy_cn = balanced_accuracy_cn + acc4
            b_cn.append(acc4)

        total_balanced_accuracy = total_balanced_accuracy + balance_accuracy / 10
        total_balanced_accuracy_k = total_balanced_accuracy_k + balanced_accuracy_k / 10
        total_balanced_accuracy_bs = total_balanced_accuracy_bs + balanced_accuracy_bs / 10
        total_balanced_accuracy_cn = total_balanced_accuracy_cn + balanced_accuracy_cn / 10

        counter = counter + 1
        print(counter)

    return [total_balanced_accuracy / 10,
            total_balanced_accuracy_k / 10,
            total_balanced_accuracy_bs / 10,
            total_balanced_accuracy_cn / 10, b_full, b_kaiser, b_bs, b_cn]


def main():
    # read dimensions.csv to read number of components
    dimensions = pd.read_csv('dimensions.csv')

    print('Banknote.mat')
    file = 'Banknote.mat'
    [b, b_k, b_bs, b_cn, sample_full, sample_kaiser, sample_bs, sample_cn] = balanced_accuracy(file, dimensions, 17)
    accuracy = pd.DataFrame({
        'D': sample_full,
        'Kaiser': sample_kaiser,
        'Broken': sample_bs,
        'Condition': sample_cn
    })
    accuracy.to_csv('knn_' + file, index=False)
    print("b : ", round(b, 4))
    print("b_k : ", round(b_k, 4))
    print("b_bs : ", round(b_bs, 4))
    print("b_cn : ", round(b_cn, 4))

    print('blood.mat')
    file = 'blood.mat'
    [b, b_k, b_bs, b_cn, sample_full, sample_kaiser, sample_bs, sample_cn] = balanced_accuracy(file, dimensions, 23)
    accuracy = pd.DataFrame({
        'D': sample_full,
        'Kaiser': sample_kaiser,
        'Broken': sample_bs,
        'Condition': sample_cn
    })
    accuracy.to_csv('knn_' + file, index=False)
    print("b : ", round(b, 4))
    print("b_k : ", round(b_k, 4))
    print("b_bs : ", round(b_bs, 4))
    print("b_cn : ", round(b_cn, 4))

    print('breastCancer.mat')
    file = 'breastCancer.mat'
    [b, b_k, b_bs, b_cn, sample_full, sample_kaiser, sample_bs, sample_cn] = balanced_accuracy(file, dimensions, 5)
    accuracy = pd.DataFrame({
        'D': sample_full,
        'Kaiser': sample_kaiser,
        'Broken': sample_bs,
        'Condition': sample_cn
    })
    accuracy.to_csv('knn_' + file, index=False)
    print("b : ", round(b, 4))
    print("b_k : ", round(b_k, 4))
    print("b_bs : ", round(b_bs, 4))
    print("b_cn : ", round(b_cn, 4))

    print('climate.mat')
    file = 'climate.mat'
    [b, b_k, b_bs, b_cn, sample_full, sample_kaiser, sample_bs, sample_cn] = balanced_accuracy(file, dimensions, 5)
    accuracy = pd.DataFrame({
        'D': sample_full,
        'Kaiser': sample_kaiser,
        'Broken': sample_bs,
        'Condition': sample_cn
    })
    accuracy.to_csv('knn_' + file, index=False)
    print("b : ", round(b, 4))
    print("b_k : ", round(b_k, 4))
    print("b_bs : ", round(b_bs, 4))
    print("b_cn : ", round(b_cn, 4))

    print('Cryotherapy.mat')
    file = 'Cryotherapy.mat'
    [b, b_k, b_bs, b_cn, sample_full, sample_kaiser, sample_bs, sample_cn] = balanced_accuracy(file, dimensions, 7)
    accuracy = pd.DataFrame({
        'D': sample_full,
        'Kaiser': sample_kaiser,
        'Broken': sample_bs,
        'Condition': sample_cn
    })
    accuracy.to_csv('knn_' + file, index=False)
    print("b : ", round(b, 4))
    print("b_k : ", round(b_k, 4))
    print("b_bs : ", round(b_bs, 4))
    print("b_cn : ", round(b_cn, 4))

    print('diabetic.mat')
    file = 'diabetic.mat'
    [b, b_k, b_bs, b_cn, sample_full, sample_kaiser, sample_bs, sample_cn] = balanced_accuracy(file, dimensions, 29)
    accuracy = pd.DataFrame({
        'D': sample_full,
        'Kaiser': sample_kaiser,
        'Broken': sample_bs,
        'Condition': sample_cn
    })
    accuracy.to_csv('knn_' + file, index=False)
    print("b : ", round(b, 4))
    print("b_k : ", round(b_k, 4))
    print("b_bs : ", round(b_bs, 4))
    print("b_cn : ", round(b_cn, 4))


if __name__ == "__main__":
    main()

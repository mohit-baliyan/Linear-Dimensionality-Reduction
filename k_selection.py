import os
from pca import PCA
from knn import KNN
import pandas as pd
from scipy import stats
from scipy.io import loadmat
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import balanced_accuracy_score


def select_best_k(file, dimensions):

    # load database file
    data = loadmat('Databases/' + file)
    X = data['X']
    y = data['Y']

    # standardise X
    X = stats.zscore(X)

    # initialize PCA
    pca = PCA()

    # read dimensions.csv and transform X_train, X_test
    dim = dimensions.loc[dimensions['Databases'] == file]
    X_kaiser = pca.transformation(X, dim["PCA-K"].values[0])
    X_bs = pca.transformation(X, dim["PCA-BS"].values[0])
    X_cn = pca.transformation(X, dim["PCA-CN"].values[0])

    # store results for each models in list with number of neighbors e.g (0.82, 3)
    balanced_accuracies_kaiser = []
    balanced_accuracies_bs = []
    balanced_accuracies_cn = []

    for i in range(1, 30):

        # initialize leave one out cross validation
        loo = LeaveOneOut()

        # leave one out cross validation
        y_kaiser_predict = []
        y_bs_predict = []
        y_cn_predict = []
        y_true = []
        for m, n in loo.split(X_kaiser, y):

            # on kaiser
            X_train_kaiser, X_test_kaiser = X_kaiser[m, :], X_kaiser[n, :]
            y_train, y_test = y[m], y[n]
            model_kaiser = KNN(k=i)
            model_kaiser.fit(X_train_kaiser, y_train)
            y_kaiser_predict.append(model_kaiser.predict(X_test_kaiser)[0])

            # on broken stick
            X_train_bs, X_test_bs = X_bs[m, :], X_bs[n, :]
            y_train, y_test = y[m], y[n]
            model_bs = KNN(k=i)
            model_bs.fit(X_train_bs, y_train)
            y_bs_predict.append(model_bs.predict(X_test_bs)[0])

            # on conditional number
            X_train_cn, X_test_cn = X_cn[m, :], X_cn[n, :]
            y_train, y_test = y[m], y[n]
            model_cn = KNN(k=i)
            model_cn.fit(X_train_cn, y_train)
            y_cn_predict.append(model_cn.predict(X_test_cn)[0])

            y_true.append(y_test[0])

        balanced_accuracies_kaiser.append((balanced_accuracy_score(y_true, y_kaiser_predict), i))
        balanced_accuracies_bs.append((balanced_accuracy_score(y_true, y_bs_predict), i))
        balanced_accuracies_cn.append((balanced_accuracy_score(y_true, y_cn_predict), i))

    # return value of number of neighbors for best models for all three configuration of X
    return max(balanced_accuracies_kaiser)[1], max(balanced_accuracies_bs)[1], max(balanced_accuracies_cn)[1]


def select_best_k_database():
    # read files from folder Databases
    files = os.listdir('./Databases/')

    # read dimensions.csv to read number of components for each database
    dimensions = pd.read_csv('dimensions.csv')

    # initialize lists to store best k after transform X by each component selection methodology
    Databases = []
    k_K = []
    k_BS = []
    k_CN = []

    for file in files:
        Databases.append(file)
        k_k, k_bs, k_cn = select_best_k(file, dimensions)
        k_K.append(k_k)
        k_BS.append(k_bs)
        k_CN.append(k_cn)

    # create dictionary, then convert into pandas dataframe to further save as a csv file
    dictionary = {

        'Databases': Databases,
        'k_K': k_K,
        'k_BS': k_BS,
        'k_CN': k_CN

    }

    df = pd.DataFrame(dictionary)
    df.to_csv('k.csv')


def main():
    # select best k for each database
    select_best_k_database()


if __name__ == "__main__":
    main()

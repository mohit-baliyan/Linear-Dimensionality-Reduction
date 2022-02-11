import os
import pandas as pd
from knn import KNN
from scipy.io import loadmat
from scipy.stats import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import balanced_accuracy_score


def best_K(file):

    # load database
    data = loadmat('Databases/' + file)
    X = data['X']
    y = data['Y']

    # standardise
    X = stats.zscore(X)

    best_K = 0
    best_score = 0

    # testing
    count  = 0

    # select best K
    for i in range(1, 30, 2):

        # initialize leave one out cross validation object
        cv = LeaveOneOut()

        # enumerate splits
        y_true, y_pred = list(), list()

        for train_ix, test_ix in cv.split(X):

            # split data
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]

            print(X_test)

            # fit model
            model = KNN(k=i)
            model.fit(X_train, y_train)

            # evaluate model
            yhat = model.predict(X_test)

            # store
            y_true.append(y_test[0])
            y_pred.append(yhat[0])

            # test nested loop
            print("fold executed")
            print(count)
            count = count + 1

        # calculate accuracy
        score = balanced_accuracy_score(y_true, y_pred)

        # select best score and best K
        if score > best_score:
            best_score = score
            best_K = i

        # test outer loop
        print("k executed")
        print("next i for k")

    # delete caches
    del data, X, y, best_score, cv, y_true, y_pred, train_ix, test_ix, X_train, X_test, y_train, y_test, model, \
        yhat, score

    return best_K


def databases():

    # # read files
    # files = os.listdir('./Folds-Databases/')
    #
    # # initialize lists to store results
    # Databases = []
    # K = []
    #
    # for file in files:
    #
    #     Databases.append(file)
    #     K.append(best_K(file))
    #
    # # create dictionary, then convert into pandas dataframe to further save as a csv file
    # dictionary = {
    #
    #     'Databases': Databases,
    #     'K': K,
    #
    # }
    #
    # df = pd.DataFrame(dictionary)
    # df.to_csv('K_selection1.csv')

    print(best_K('minboone.mat'))
    # print(best_K('skin.mat'))
    # print(best_K('EggEyeState.mat'))
    # print(best_K('telescope.mat'))
    # print(best_K('HTRU2.mat'))
    # print(best_K('LfW_faces.mat'))



def main():

    # K selection
    databases()


if __name__ == "__main__":
    main()
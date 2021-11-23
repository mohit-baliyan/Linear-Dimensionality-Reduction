import os
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


# create folds and write to HDD
def save_folds(X, y):
    # create dataframe df and later join vectors in X and vector y
    df = pd.DataFrame(X)
    df['y'] = y

    # create directory for saving folds
    os.mkdir('folds')

    # initialize Stratified K fold cross validation
    skf = StratifiedKFold(n_splits=10, shuffle=True)

    # looping over folds
    fold_no = 1
    for train_index, val_index in skf.split(df, y):
        # slicing
        train = df.loc[train_index, :]
        val = df.loc[val_index, :]

        # reset index for each fold data
        train.reset_index(inplace=True)
        val.reset_index(inplace=True)

        # save to HDD
        train.to_csv('folds/train_fold_' + str(fold_no) + '.csv', index=False)
        val.to_csv('folds/val_fold_' + str(fold_no) + '.csv', index=False)

        fold_no += 1


# return the average of balanced accuracy after running 10 times with 10 fold stratified cross-validation
def calculate_accuracy(X, y, model):
    # create folds and write to HDD
    save_folds(X, y)

    # outer loop to calculate the balanced accuracy 10 times
    balanced_accuracies = []
    for i in range(0, 10):
        balanced_accuracy_10_folds = []

        # inner loop for 10 fold stratified cross validation
        for j in range(1, 11):
            # read folds from csv and convert into numpy array
            df_train = pd.read_csv('folds/train_fold_' + str(j) + '.csv', index_col=0)
            df_test = pd.read_csv('folds/val_fold_' + str(j) + '.csv', index_col=0)
            df_train = df_train.values
            df_test = df_test.values

            # slicing into X_train, y_train, X_test and y_test
            X_train = df_train[:, :-1]
            y_train = df_train[:, -1]
            X_test = df_test[:, :-1]
            y_test = df_test[:, -1]

            # train model and predict X_test
            model.fit(X_train, y_train)
            balanced_accuracy_10_folds.append(balanced_accuracy_score(y_test, model.predict(X_test)))

        balanced_accuracies.append(np.mean(balanced_accuracy_10_folds))

    # delete directory
    shutil.rmtree('folds')

    return np.mean(balanced_accuracies)

import os
import warnings
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold


# create folds and write their indices to HDD
def save_folds(x, y, database):
    # create dataframe df and later join vectors in x and vector y
    df = pd.DataFrame(x)
    df['y'] = y

    # create directory for saving folds
    os.mkdir('Folds-Databases/' + database)

    # initialize Stratified K fold cross validation
    skf = StratifiedKFold(n_splits=10, shuffle=False)

    # looping over folds
    fold_no = 1
    for train_index, test_index in skf.split(df, y):
        # slicing and save indices  to HDD
        np.savetxt('Folds-Databases/' + database + '/train_fold_' + str(fold_no) + '.txt', train_index)
        np.savetxt('Folds-Databases/' + database + '/test_fold_' + str(fold_no) + '.txt', test_index)

        fold_no = fold_no + 1


def main():
    warnings.filterwarnings("ignore")

    # load one database at a time
    files = os.listdir('./Databases/')
    for file in files:

        data = loadmat('./Databases/' + file)
        x = data['X']
        y = data['Y']

        # call save_folds(x, y,  file ) if folds does not exist
        if not os.path.isdir('Folds-Databases/' + file):
            save_folds(x, y, file)


if __name__ == "__main__":
    main()

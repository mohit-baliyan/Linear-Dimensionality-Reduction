from pca import PCA
from knn import KNN
import pandas as pd
from scipy.io import loadmat
from scipy.stats import stats
from balanced_accuracy import calculate_accuracy


def main():
    # read dimensions
    df = pd.read_csv('frames.csv')

    # load database
    database = 'telescope.mat'
    data = loadmat('./Databases/' + database)
    X = data['X']
    y = data['Y']

    # standardise data
    X = stats.zscore(X)

    # initialize KNN model
    knn = KNN(3)

    # without transformation
    accuracy_original = calculate_accuracy(X, y, knn, database[:-4]+'O')

    # initialize pca
    pca = PCA()

    # after kaiser rule
    accuracy_kaiser = calculate_accuracy(pca.transformation(X, df[df['Databases'] == database]['PCA-K'].item()), y,
                                         knn, database[:-4]+'K')

    # after broken stick
    accuracy_bs = calculate_accuracy(pca.transformation(X, df[df['Databases'] == database]['PCA-BS'].item()), y,
                                     knn, database[:-4]+'BS')

    # after conditional number
    accuracy_cn = calculate_accuracy(pca.transformation(X, df[df['Databases'] == database]['PCA-CN'].item()), y, knn,
                                     database[:-4]+'CN')

    print(database, round(accuracy_original, 2), round(accuracy_kaiser, 2), round(accuracy_bs, 2),
          round(accuracy_cn, 2))


if __name__ == "__main__":
    main()

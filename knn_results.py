import os
from pca import PCA
from knn import KNN
import pandas as pd
from scipy.io import loadmat
from scipy.stats import stats
from balanced_accuracy import calculate_accuracy


def main():
    datasets = []
    accuracy_original = []
    accuracy_kaiser = []
    accuracy_broken = []
    accuracy_cn = []
    df = pd.read_csv('frames.csv')
    databases = os.listdir('./Databases/')
    for database in databases:
        # load dataset from Databases
        data = loadmat('./Databases/' + database)
        X = data['X']
        y = data['Y']
        # standardise data
        X = stats.zscore(X)
        # initialize KNN model
        knn = KNN(3)
        accuracy_original.append(calculate_accuracy(X, y, knn))
        pca = PCA()
        # for kaiser rule
        accuracy_kaiser.append(
            calculate_accuracy(pca.transformation(X, df[df['Databases'] == database]['PCA-K'].item()), y, knn))
        # for broken stick
        accuracy_broken.append(
            calculate_accuracy(pca.transformation(X, df[df['Databases'] == database]['PCA-BS'].item()), y, knn))
        # for conditional number
        accuracy_cn.append(calculate_accuracy(pca.transformation(X, df[df['Databases'] == database]['PCA-CN'].item()),
                                              y, knn))
        datasets.append(database)
    dictionary = {'Databases': datasets, 'Accuracy-O': accuracy_original, 'Accuracy-K': accuracy_kaiser,
                  'Accuracy-BS': accuracy_broken, 'Accuracy-CN': accuracy_cn}

    # convert dictionary into pandas dataframe and save into csv
    acc = pd.DataFrame(dictionary)
    acc.to_csv('performance_knn.csv')


if __name__ == "__main__":
    main()

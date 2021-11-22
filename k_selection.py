import pandas as pd
from knn import KNN
from scipy.io import loadmat
from scipy.stats import stats
from pca import PCA, SelectionMethods
from balanced_accuracy import calculate_accuracy


def main():
    # load dataset
    data = loadmat('./Databases/Musk.mat')
    X = data['X']
    y = data['Y']

    # standardise data
    X = stats.zscore(X)

    # apply pca with conditional number
    pca = PCA()
    method = SelectionMethods(X)
    conditional_comps = method.conditional_number()
    X_transform = pca.transformation(X, conditional_comps)

    # knn modelling and save results for different values of k
    k = []
    accuracy = []
    for i in range(1, 30):
        knn = KNN(i)
        k.append(i)
        accuracy.append(calculate_accuracy(X_transform, y, knn))

    # create dictionary, then convert into pandas dataframe to further save as a csv file
    dictionary = {'K': k, 'Accuracy': accuracy}
    df = pd.DataFrame(dictionary)
    df.to_csv("neighbors.csv")


if __name__ == "__main__":
    main()

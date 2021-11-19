import pandas as pd
from knn import KNN, magic
from scipy.io import loadmat
from scipy.stats import stats
from componentSelectionPCA import PCA, SelectionMethods


def main():
    data = loadmat('./Databases/Musk.mat')
    X = data['X']
    # standardise data
    X = stats.zscore(X)
    y = data['Y']
    pca = PCA()
    method = SelectionMethods(X)
    conditional_comps = method.conditional_number()
    X_transform = pca.transformation(X, conditional_comps)
    K = []
    Accuracy = []
    for i in range(1, 30):
        knn = KNN(i)
        K.append(K)
        Accuracy.append(magic(X_transform, y, knn))
    dictionary = {'K': K, 'Accuracy': Accuracy}
    df = pd.DataFrame(dictionary)
    df.to_csv("neighbors.csv")


if __name__ == "__main__":
    main()

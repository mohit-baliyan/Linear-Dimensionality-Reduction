import os
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import mode
import scipy.stats as stats
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from componentSelectionPCA import PCA


# K Nearest Neighbors Classification
class KNN:

    def __init__(self, k):
        self.k = k

    # Function to store training set
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        # no_of_training_examples, no_of_features
        self.m, n = self.X_train.shape

    # Function for prediction
    def predict(self, X_test):
        # no_of_test_examples, no_of_features
        m_test, n = X_test.shape
        # initialize Y_predict
        Y_predict = np.zeros(m_test)
        for i in range(m_test):
            x = X_test[i]
            # find the k nearest neighbors from current test example
            neighbors = np.zeros(self.k)
            neighbors = self.find_neighbors(x)
            # most frequent class in k neighbors
            Y_predict[i] = mode(neighbors)[0][0]
        return Y_predict

    # Function to find the k nearest neighbors to current test example
    def find_neighbors(self, x):
        # calculate all the euclidean distances between current test example x and training set X_train
        euclidean_distances = np.zeros(self.m)
        for i in range(self.m):
            d = self.euclidean(x, self.X_train[i])
            euclidean_distances[i] = d
        # sort Y_train according to euclidean_distance_array and store into Y_train_sorted
        ids = euclidean_distances.argsort()
        Y_train_sorted = self.Y_train[ids]
        return Y_train_sorted[:self.k]

    # Function to calculate euclidean distance
    def euclidean(self, x, xm):
        return np.sqrt(np.sum(np.square(x - xm)))


# return the average of balanced accuracy after running 10 times with 10 fold stratified cross-validation
def magic(X, y, model):
    # outer loop to calculate the balanced accuracy 10 times
    balanced_accuracies = []
    for i in range(0, 10):
        # shuffle X, y before Splitting
        shuffle(X, y)
        fold = StratifiedKFold(n_splits=10, shuffle=True)
        balanced_accuracy_10_folds = []
        # inner loop for 10 fold stratified cross validation
        for train_index, test_index in fold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            balanced_accuracy_10_folds.append(balanced_accuracy_score(y_test, model.predict(X_test)))
        balanced_accuracies.append(np.mean(balanced_accuracy_10_folds))

    return np.mean(balanced_accuracies)


def main():
    accuracy_original = []
    accuracy_kaiser = []
    accuracy_broken = []
    accuracy_conditional = []
    databases = os.listdir('./Databases/')
    for database in databases:
        data = loadmat('./Databases/' + database)
        X = data['X']
        y = data['Y']
        df = pd.read_csv('frames.csv')
        # standardise data
        X = stats.zscore(X)
        # initialize KNN model
        knn = KNN(3)
        accuracy_original.append(magic(X, y, knn))
        pca = PCA()
        row = df.loc[df['Databases'] == database]
        # for kaiser rule
        accuracy_kaiser.append(magic(pca.transformation(X, int(row['PCA-K'])), y, knn))
        # for broken stick
        accuracy_broken.append(magic(pca.transformation(X, int(row['PCA-BS'])), y, knn))
        # for conditional number
        accuracy_conditional.append(magic(pca.transformation(X, int(row['PCA-CN'])), y, knn))
        dictionary = {'Accuracy-O': accuracy_original, 'Accuracy-K': accuracy_kaiser,
                      'Accuracy-BS': accuracy_broken, 'Accuracy-CN': accuracy_conditional}
        print('Running..')
    acc = pd.DataFrame(dictionary)
    acc.to_csv('performance-knn.csv')


if __name__ == "__main__":
    main()

import os
import pandas as pd
from scipy.io import loadmat
from scipy.stats import stats
from pca import SelectionMethods


def main():

    # create list of databases
    files = os.listdir('./Databases/')

    # initialize lists to store each database name, no of attributes, cases and respective dimensions
    databases = []
    original_attributes = []
    cases = []
    kaiser_comps = []
    broken_comps = []
    conditional_comps = []

    for file in files:

        # load one database at a time
        data = loadmat('./Databases/' + file)
        X = data['X']

        # standardise data
        X = stats.zscore(X)

        # apply selection methods for calculating respective dimensions
        methods = SelectionMethods(X)

        # for kaiser rule
        kaiser_comps.append(methods.kaiser_rule())

        # for broken stick
        broken_comps.append(methods.broken_stick())

        # for conditional number
        conditional_comps.append(methods.conditional_number())

        # original dimensions
        (m, n) = X.shape
        databases.append(file)
        cases.append(m)
        original_attributes.append(n)

    # create dictionary, then convert into pandas dataframe to further save as a csv file
    dictionary = {'Databases': databases, 'Cases': cases, 'Attributes': original_attributes, 'PCA-K': kaiser_comps,
                  'PCA-BS': broken_comps, 'PCA-CN': conditional_comps}

    df = pd.DataFrame(dictionary)
    df.to_csv('dimensions.csv')


if __name__ == "__main__":
    main()
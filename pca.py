import numpy as np
from numpy.linalg import eig


class PCA:

    # function to calculate eigen values and eigen vector for any matrix
    def eig_vector(self, x):

        # centralize
        mean = np.mean(x, 0)
        x_stand = x - mean

        # calculate correlation matrix
        x_cov = np.corrcoef(np.transpose(x_stand))

        # find the eigenvalues and eigenvectors
        e, V = eig(x_cov)

        # sort eigen vector according to eigen values
        idx = np.argsort(-e)

        # selection
        e = e[idx]
        V = V[:, idx]

        return e, V

    # projection of X
    def transformation(self, x, no_of_components):

        e, V = self.eig_vector(x)
        p = V[:, : no_of_components]

        # project the original dataset
        mean = np.mean(x, 0)
        x_stand = x - mean
        x_transform = np.dot(x_stand, p)

        return x_transform


class SelectionMethods:

    def __init__(self, x):
        self.x = x

    # function return number of components
    def conditional_number(self):

        pca = PCA()
        e, V = pca.eig_vector(self.x)

        # selection
        e_max = e[0]
        condition = e_max / 10
        no_of_components = np.count_nonzero(e > condition)

        if no_of_components == 0:
            return 1
        else:
            return no_of_components

    # function return number of components
    def kaiser_rule(self):

        pca = PCA()
        e, V = pca.eig_vector(self.x)

        # selection
        return np.argmax(e < 1)

    # function return number of components
    def broken_stick(self):

        pca = PCA()
        e, V = pca.eig_vector(self.x)

        # calculate the proportional variance
        prop_var = e / sum(e)

        # calculate the expected length of the k-th longest segment
        p = np.size(e)
        g = np.zeros(p)

        k = 0
        while k < p:
            i = k
            while i < p:
                g[k] = g[k] + (1 / (i + 1))
                i = i + 1
            k = k + 1
        g = g / p

        '''
        In the Broken-Stick model, the individual percentages of variance of the components are compared with the
        values expected from the “broken stick” distribution. The two distributions are compared element-by-element,
        and first value d + 1 where the expected values larger than the observed value determines the dimension. 
        '''
        no_of_components = np.argmax(prop_var < g)

        if no_of_components == 0:
            return 1
        else:
            return no_of_components
#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os

import pandas as pd

import numpy as np

from scipy.io import loadmat

from sklearn.metrics import balanced_accuracy_score

from sklearn.model_selection import StratifiedKFold, train_test_split

import warnings

warnings.filterwarnings("ignore")

import scipy.stats as stats


# In[12]:


# Logistic Regression

class LogitRegression() :
    
    
    
    def __init__( self, learning_rate, iterations ) :        
        
        self.learning_rate = learning_rate        
        
        self.iterations = iterations
        
        
        
          
    # Function for model training   
    
    def fit( self, X, Y ) :        
        
        # no_of_training_examples, no_of_features        
        
        self.m, self.n = X.shape        
        
        # weight initialization        
        
        self.W = np.zeros( self.n )        
        
        self.b = 0        
        
        self.X = X        
        
        self.Y = Y
          
        # gradient descent learning
                  
        for i in range( self.iterations ) :            
            
            self.update_weights()            
        
        return self
      
    
    
    # Helper function to update weights in gradient descent
      
    def update_weights( self ) :           
        
        A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
          
        # calculate gradients        
        
        tmp = ( A - self.Y.T )        
        
        tmp = np.reshape( tmp, self.m  )        
        
        dW = np.dot( self.X.T, tmp ) / self.m         
        
        db = np.sum( tmp ) / self.m 
          
        # update weights    
        
        self.W = self.W - self.learning_rate * dW    
        
        self.b = self.b - self.learning_rate * db
          
        return self
      
    
    
    # Hypothetical function  h( x ) 
      
    def predict( self, X ) :    
        
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )          
        
        return Z


# In[13]:


def threshold_selection( z, y  ) :
    
    
    # total number of examples 
    
    total = np.size( z )
    
    
    # rehshape y into 1 D as of z
    
    y = np.reshape( y, ( total ) )
    
    
    
    # define set of unique values with all possible points
    
    thres = np.unique( z )
    
    # add all borders
    
    thres = ( thres[1:] + thres[:-1] ) / 2
    
    
    
    
    # selecting threshold with best error
    
    besterror = total
    
    bestthres = 0
    
    for t in thres :
        
        y_hat = np.where( z > t, 1, 0)
        
        error = sum( y_hat != y )
        
        if( error < besterror ) :
            
            besterror = error
            
            bestthres = t
    
    return bestthres


# In[ ]:


files = os.listdir( './Databases/' )

Databases = []

Thresholds = []


for file in files :
    
    
    # read one database at a time
    
    data = loadmat( './Databases/' + file )
    
    X = data['X']    
    
    # standarise data
    
    X = stats.zscore( X )
    
    

    y = data['Y']
    
    ( m , n ) = X.shape
    

    try :
            
        model = LogitRegression( learning_rate = 0.01, iterations = 500 )
        
        model.fit( X, y )
        
        z = model.predict( X )
        
        bestthres = threshold_selection( z, y )
        
        Databases.append( file )
        
        Thresholds.append( bestthres )
            
    except :
        
            
        continue


# In[ ]:


Thres = { 'Databases' : Databases, 'Thresholds' : Thresholds }

Thres = pd.DataFrame( Thres )

Thres


# In[ ]:


Thres.to_csv( 'Thres.csv' )

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import mode
import scipy.stats as stats


# K Nearest Neighbors Classification
class KNN():

    def __init__(self, k):
        self.k = k

    # Function to store training set
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

        # no_of_training_examples, no_of_features
        self.m, self.n = X_train.shape

    # Function for prediction
    def predict(self, X_test):
        self.X_test = X_test
        # no_of_test_examples, no_of_features
        self.m_test, self.n = X_test.shape
        # initialize Y_predict
        Y_predict = np.zeros(self.m_test)
        for i in range(self.m_test):
            x = self.X_test[i]
            # find the K nearest neighbors from current test example
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
        inds = euclidean_distances.argsort()
        Y_train_sorted = self.Y_train[inds]
        return Y_train_sorted[:self.k]

    # Function to calculate euclidean distance
    def euclidean(self, x, x_train):
        return np.sqrt(np.sum(np.square(x - x_train)))


# ## Logistic Regression

# In[36]:


# # Logistic Regression

class LogitRegression():

    def __init__(self, learning_rate, iterations, threshold):
        self.learning_rate = learning_rate

        self.iterations = iterations

        self.threshold = threshold

    # Function for model training

    def fit(self, X, Y):
        # no_of_training_examples, no_of_features

        self.m, self.n = X.shape

        # weight initialization

        self.W = np.zeros(self.n)

        self.b = 0

        self.X = X

        self.Y = Y

        # gradient descent learning

        for i in range(self.iterations):
            self.update_weights()

        return self

    # Helper function to update weights in gradient descent

    def update_weights(self):
        A = 1 / (1 + np.exp(- (self.X.dot(self.W) + self.b)))

        # calculate gradients

        tmp = (A - self.Y.T)

        tmp = np.reshape(tmp, self.m)

        dW = np.dot(self.X.T, tmp) / self.m

        db = np.sum(tmp) / self.m

        # update weights

        self.W = self.W - self.learning_rate * dW

        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function  h( x )

    def predict(self, X):
        Z = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))

        Y = np.where(Z > self.threshold, 1, 0)

        return Y


# # Selecting Components for Databases

# In[37]:


files = os.listdir('./Databases/')

# Initialize lists to store each database name, no of attributes, cases and their respective dimensions

Databases = []

Attributes = []

Cases = []

PCA_K = []

PCA_BS = []

PCA_CN = []

for file in files:

    # read one database at a time

    data = loadmat('./Databases/' + file)

    X = data['X']

    # standarise data

    X = stats.zscore(X)

    y = data['Y']

    try:

        # no. of components selected by kaiser, broken stick and condtional number

        # kaiser rule

        no_of_components_kaiser_rule = kaiser_rule(X)

        PCA_K.append(no_of_components_kaiser_rule)

        # broken stick

        no_of_components_broken_stick_rule = broken_stick(X)

        PCA_BS.append(no_of_components_broken_stick_rule)

        # conditional number

        no_of_components_conditonal_number = conditional_number(X)

        PCA_CN.append(no_of_components_conditonal_number)

        # get shapes and append them in their lists

        (m, n) = X.shape

        Databases.append(file)

        Attributes.append(n)

        Cases.append(m)



    except:

        continue

# In[38]:


Component_Selection = {'Databases': Databases, 'Attributes': Attributes, 'Cases': Cases, 'PCA-K': PCA_K,
                       'PCA-BS': PCA_BS,

                       'PCA-CN': PCA_CN

                       }

Component_Selection = pd.DataFrame(Component_Selection)

Component_Selection.to_csv('Component_Selection.csv')

Component_Selection


# # Modelling and Balanced Accuracy calculation 10 times

# In[39]:


# return the average of balanced accuracy after running 10 times with 10 fold stratified cross-validation

def magic(X, y, model):
    # outer loop to calculate the balanced accuracy 10 times

    balanced_accuracies = []

    for i in range(0, 10):

        # shuffle X, y before Splitting

        shuffle(X, y)

        skfold = StratifiedKFold(n_splits=10, shuffle=True)

        balanced_accuracy_K_folds = []

        # inner loop for 10 fold stratified cross validation

        for train_index, test_index in skfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]

            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)

            balanced_accuracy_K_folds.append(balanced_accuracy_score(y_test, model.predict(X_test)))

        balanced_accuracies.append(np.mean(balanced_accuracy_K_folds))

    return np.mean(balanced_accuracies)


# # KNN Results

# In[40]:


Thres = pd.read_csv('Thres.csv')

Component_Selection = pd.read_csv('Component_Selection.csv')

Databases = Thres["Databases"]

# In[41]:


Databases = Databases[:10]

# In[42]:


BA_K = []

BA_BS = []

BA_CN = []

for file in Databases:
    # Get X and Y

    data = loadmat('Databases/' + file)

    X = data['X']

    # standarise data

    X = stats.zscore(X)

    y = data['Y']

    # select row from Component_Selection dataframe

    x = Component_Selection.loc[Component_Selection['Databases'] == file]

    # For Kaiser-rule

    no_of_components_kaiser_rule = x["PCA-K"].values[0]

    pca = PCA()

    X_kaiser_rule = pca.transformation(X, no_of_components_kaiser_rule)

    knn = K_Nearest_Neighbors_Classifier(K=3)

    BA_K.append(magic(X_kaiser_rule, y, knn))

    # For Broken Stick

    no_of_components_broken_stick = x["PCA-BS"].values[0]

    pca = PCA()

    X_broken_stick = pca.transformation(X, no_of_components_broken_stick)

    knn = K_Nearest_Neighbors_Classifier(K=3)

    BA_BS.append(magic(X_broken_stick, y, knn))

    # For conditional number

    no_of_components_conditonal_number = x["PCA-CN"].values[0]

    pca = PCA()

    X_conditional_number = pca.transformation(X, no_of_components_conditonal_number)

    knn = K_Nearest_Neighbors_Classifier(K=3)

    BA_CN.append(magic(X_conditional_number, y, knn))

# In[43]:

Databases = Databases.to_list()

# In[44]:


KNN_Results = {'Databases': Databases, 'BA_K': BA_K, 'BA_BS': BA_BS, 'BA_CN': BA_CN}

KNN_Results = pd.DataFrame(KNN_Results)

KNN_Results.to_csv('KNN_Results.csv')

# In[45]:


KNN_Results

# # Logistic Regression Results

# In[46]:


BA_K = []

BA_BS = []

BA_CN = []

for file in Databases:

    # Get X and Y

    data = loadmat('Databases/' + file)

    X = data['X']

    # standarise data

    X = stats.zscore(X)

    y = data['Y']

    # select row from Component_Selection dataframe

    x = Component_Selection.loc[Component_Selection['Databases'] == file]

    # select row from Component_Selection dataframe

    z = Thres.loc[Thres["Databases"] == file]

    best_thres = z["Thresholds"].values[0]

    if (math.isnan(best_thres)):
        best_thres = 0.5

    # For Kaiser-rule

    no_of_components_kaiser_rule = x["PCA-K"].values[0]

    pca = PCA()

    X_kaiser_rule = pca.transformation(X, no_of_components_kaiser_rule)

    model = LogitRegression(0.01, 500, best_thres)

    BA_K.append(magic(X_kaiser_rule, y, model))

    # For Broken Stick

    no_of_components_broken_stick = x["PCA-BS"].values[0]

    pca = PCA()

    X_broken_stick = pca.transformation(X, no_of_components_broken_stick)

    model = LogitRegression(0.01, 500, best_thres)

    BA_BS.append(magic(X_broken_stick, y, model))

    # For conditional number

    no_of_components_conditonal_number = x["PCA-CN"].values[0]

    pca = PCA()

    X_conditional_number = pca.transformation(X, no_of_components_conditonal_number)

    model = LogitRegression(0.01, 500, best_thres)

    BA_CN.append(magic(X_conditional_number, y, model))

# In[47]:

Logistic_Results = {'Databases': Databases, 'BA_K': BA_K, 'BA_BS': BA_BS, 'BA_CN': BA_CN}

Logistic_Results = pd.DataFrame(Logistic_Results)

# Logistic_Results.to_csv(Logistic_Results.csv')

# In[48]:


Logistic_Results

# In[50]:


# while( True ) :

#     code()

#     eat()

#     sleep()


# # Fisher LDA Results





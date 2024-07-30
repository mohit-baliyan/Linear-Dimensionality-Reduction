# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:17:45 2024
Recalculation of data for paper

@author: em322
"""

# pylint: disable=C0103

import os
import os.path
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def databseConvert(name):
    '''
    Get database databases\name+".mat", standardise data matrix X and convert
    to pkl format with recording of tuple (X,y)

    Parameters
    ----------
    name : string
        Name of database without extension.

    Returns
    -------
    None.

    '''
    data = loadmat('databases/' + name + '.mat')
    X = data['X']
    y = data['Y']
    # Standardise data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # Save results.
    with open('databases/' + name + '.pkl', 'wb') as fil:
        pickle.dump((X, y), fil, pickle.HIGHEST_PROTOCOL)

def foldsForming(name):
    '''
    foldsForming creates folds for 10 times 10 fold cross validation.
    Folds must be the same for all dimasions and all classifiers.
    Folds are stored to folds/name/ folder, where name is name of database.

    Parameters
    ----------
    name : string
        Name of database without extension.

    Returns
    -------
    None.

    '''
    # Create folder for data
    if not os.path.exists('Folds'):
        os.mkdir('Folds')
    if not os.path.exists('Folds/' + name):
        os.mkdir('Folds/' + name)
    # Load database
    X, y = readDatabase(name)

    # Create folds and save infromation to files
    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
    k = 0
    for train, test in skf.split(X, y):
        # slicing and save indices  to HDD
        np.savetxt(f'Folds/{name}/train_{k:03d}.txt', train, fmt='%d')
        np.savetxt(f'Folds/{name}/test_{k:03d}.txt', test, fmt='%d')
        k += 1

def readDatabase(name):
    '''
    readDatabase load specified database

    Parameters
    ----------
    name : string
        Name of database without extension.

    Returns
    -------
    X : 2D np.ndarray
        Data table of loaded database
    y : 1D np.ndarray
        Labels of loaded database
    '''
    with open('databases/' + name + '.pkl', 'rb') as inp:
        tmp = pickle.load(inp)
    X = tmp[0]
    y = tmp[1]
    return (X, y)

def readDimensions():
    '''
    Load file dimensions.csv with dimensions of databases

    Returns
    -------
    dimsi : 2D np.ndarray
        This table contains one row for each database. Each row contains four
        dimansions:
            0: number of attributes
            1: PCA-K
            2: PCA-BS
            3: PCA-CN
    '''
    with open('dimensions.csv', encoding="utf-8") as file:
        content = file.readlines()
    # There is 1 line of header
    content = content[1:]
    # Create tables for reading
    dimsi = np.zeros((len(content), 4), dtype=int)
    # pp is row to write
    pp = 0
    for line in content:
        tmp = line.split(",")
        for k in range(4):
            dimsi[pp, k] = int(tmp[k + 2])
        pp += 1
    return dimsi

def dimCorrelations():
    '''
    dimCorrelations calculate degree of independence of dimansions through
    paired t-test, paired Wilcoxon signed-rank test, and  Kolmogorov-Smirnov
    test.

    Returns
    -------
    resOfTest : 2D ndarray
        matrix with results of tests.

    '''
    dims = readDimensions()
    resOfTest = np.zeros((6, 3))
    pp = 0
    for k in range(3):
        for kk in range(k+1, 4):
            # ttest
            resOfTest[pp, 0] = stats.ttest_rel(dims[:, k], dims[:, kk])[1]
            resOfTest[pp, 1] = stats.wilcoxon(dims[:, k], dims[:, kk])[1]
            resOfTest[pp, 2] = stats.ks_2samp(dims[:, k], dims[:, kk])[1]
            pp += 1
    return resOfTest

# def

def kSearchKNN(name, score):
    '''
    defined the optimal k for KNN classifier. Used quality measure defined by
    score argument. Function tested all k from 1 to 500 or number of cases -1.

    Parameters
    ----------
    name : string
        Name of database to read.
    score : sklearn.metrics
        classification quality measure to optimise..

    Returns
    -------
    scors : 1D ndarray
        Specified quality measure for corresponding k: scors[k-1] for k.

    '''
    # Load database
    X, y = readDatabase(name)
    nCases = y.shape[0]
    maxk = min(nCases - 1, 500)
    # Array to save predictions
    pred = np.zeros((nCases, maxk))
    # Preliminary calculation - length of each vector
    leng = np.sum(np.square(X), axis=1)
    # Main loop Calculate required number of neighbours for each record
    for k in range(nCases):
        # multiply each row by selected one and subtract twice this value from
        # length of vector
        dist = leng - 2 * np.matmul(X, np.transpose(X[k, :])) + leng[k]
        # Sort distances to select neighbours
        ind = np.argsort(dist)
        # LOOCV required removing of the first element of array
        ind = ind[1:]
        tmp = y[ind[0:maxk]]
        # Define predictions
        tmp = np.cumsum(tmp) / np.arange(1, maxk+1)
        ind = tmp > 0.5
        pred[k, ind] = 1

    # Calculate scores
    scors = np.zeros((maxk))
    for k in range(maxk):
        scors[k] = score(y, pred[:, k])

    return scors

def readFold(name, nn):
    '''
    readFold load previously saved fold descriptions.

    Parameters
    ----------
    name : string
        Name of database to use.
    nn : int
        number of fold to read..

    Returns
    -------
    test : 1D ndarray
        List of indices for test set.
    train : 1D ndarray
        List of indices for training set.

    '''
    # read test
    with open(f'Folds/{name}/test_{nn:03d}.txt', encoding="utf-8") as file:
        content = file.readlines()
    test = np.asarray(content)
    test = test.astype(int)
    with open(f'Folds/{name}/train_{nn:03d}.txt', encoding="utf-8") as file:
        content = file.readlines()
    train = np.asarray(content)
    train = train.astype(int)
    return (test, train)

def readKScores():
    '''
    readKScores read previously saved file with kNN scores for different k.
    The first column in file contains optimal k.

    Returns
    -------
    ttt : 1D ndarray
        Selected k for all databases.

    '''
    with open('kScores.txt', encoding="utf-8") as file:
        content = file.readlines()
    # Skip header
    content = content[1:]
    # Create tables for reading
    ttt = np.zeros((len(content)), dtype=int)
    # pp is row to write
    pp = 0
    for line in content:
        tmp = line.split(",")
        ttt[pp] = int(float(tmp[0]))
        pp += 1
    return ttt

def searchThreshold(scores, score, pure):
    '''
    searchThreshold searchs the optimal threshold for quality measure defined
    in score, scores desfined in score, and labels defined in pure

    Parameters
    ----------
    scores : 1D ndarray
        scores calculated for cases by some classifier..
    score : sklearn.metrics
        classification quality measure to optimise..
    pure : 1D ndarray
        corect class labels.

    Returns
    -------
    bestTh : float
        Optimal threshold.

    '''
    # Form list of possible thresholds
    th = np.unique(scores)
    th = (th[1:] + th[:-1]) / 2
    bestTh = th[0]
    bestSc = 0
    for t in th:
        sc = score(pure, np.where(scores < t, 1, 0))
        if sc > bestSc:
            bestSc = sc
            bestTh = t
    return bestTh

def fisherDir(data, labels):
    '''
    fisherDir calculates direction of Fisher's linear discriminant for data set
    'data' and set of labels 'labels'.
    Detailed description of used model can be found in
    https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher's_linear_discriminant

    Parameters
    ----------
    data : 2D ndarray
        Data matrix.
    labels : 1D ndarrays
        Class labels.

    Returns
    -------
    direct : 1D ndarray
        DIrection of Fisher's linear discriminant.

    '''
    # Split classes
    ind = np.ravel(labels) == 0
    d0 = data[ind, :]
    d1 = data[np.logical_not(ind), :]
    # Calculate sum of covariance mareises
    matr = np.cov(d0, rowvar=False) + np.cov(d1, rowvar=False)
    # Regularize matrix
    if data.shape[1] > 1:
        matr =  + 0.001 * max(abs(np.diag(matr))) * np.eye(d0.shape[1])
        direct = np.linalg.inv(matr).dot(d0.mean(axis=0) - d1.mean(axis=0))
        direct = direct / np.sqrt(np.sum(np.square(direct)))
    else:
        direct = np.asarray([1], dtype=float)
    # Normalise direction
    return direct

def processData():
    '''
    Process data processed each database and each fold. For each fold
    processData calculated principal components on training set and then get
    specified number of PCs for three different dimensionaity reduction.

    For each fold and for each dimension (original, PCA-K, PCA-BS and PCA-CN)
    three classificatio models are formed (KNN, LR, and LDA). Results of
    calculation for each fold and database are written to separate
    txt file with name Res_dbName_nnn.txt, where dbName is name of database and
    nnn is number of fold.

    Returns
    -------
    None.

    '''
    # Load dimensions
    dims = readDimensions()
    # Load k
    ks = readKScores()
    pp = 0
    # Data base loop
    for dbf in databases:
        # To calculate only required database decomment the following if and
        # specify required number
        # if pp != 17:
        #     pp += 1
        #     continue
        X, y = readDatabase(dbf)
        # Fold loop
        for fold in range(100):
            print(dbf, fold)
            # Split test and training sets
            test, train = readFold(dbf, fold)
            XTr = X[train, :]
            XTe = X[test, :]
            yTr = y[train, :]
            yTe = y[test, :]
            # Calculate PCs
            pca = PCA(n_components=np.max(dims[pp,1:])).fit(XTr)
            cmp = pca.components_.T
            means = pca.mean_
            # Create file
            file = open(f"results/Res_{dbf}_{fold:03d}.txt", "w", encoding="utf-8")
            file.write("Dim\tClassifier\tScore\n")
            # Dimension loop
            for dim in range(4):
                if dim > 0:
                    # Project to PCs
                    xTr = np.matmul(XTr - means, cmp[:, 0:dims[pp, dim]])
                    xTe = np.matmul(XTe - means, cmp[:, 0:dims[pp, dim]])
                else:
                    xTr = XTr
                    xTe = XTe
                # Do classifiers
                # KNN
                mdl = KNeighborsClassifier(n_neighbors=ks[pp]).fit(xTr, np.ravel(yTr))
                pred = mdl.predict(xTe)
                ac = scoreToUse(yTe, pred)
                file.write(f"{dims[pp, dim]}\tKNN\t{ac}\n")

                # LR
                mdl = LogisticRegression(penalty=None).fit(xTr, np.ravel(yTr))
                # Get "probabilities"
                pred = mdl.predict_proba(xTr)
                pred = np.squeeze(pred[:, 0])
                # Define the best threshold
                th = searchThreshold(pred, scoreToUse, yTr)
                # Define prediction
                pred = mdl.predict_proba(xTe)
                pred = np.where(np.squeeze(pred[:, 0]) < th, 1, 0)
                ac = scoreToUse(yTe, pred)
                file.write(f"{dims[pp, dim]}\tLR\t{ac}\n")

                # Fisher's discriminant
                fd = fisherDir(xTr, yTr)
                # Select threshold
                th = searchThreshold(np.dot(xTr, fd), scoreToUse, yTr)
                pred = np.dot(xTe, fd)
                pred = np.where(pred < th, 1, 0)
                ac = scoreToUse(yTe, pred)
                file.write(f"{dims[pp, dim]}\tFLD\t{ac}\n")
            file.close()
        # Next database
        pp += 1

def ranksForOneDBAndClass(dbf, cls):
    '''
    ranksForOneDBAndClass load aggregated BA for database dbf and classifier cls,
    Then three types of ranks are calculated:
        1. Rank by average BA
        2. Rank according to t-test - in t-test p-value is less than 5% then
            ranks are different, otherwise values have the same rank.
        3. Rank according to Wilcoxon Signed-Rank test - in t-test p-value is
            less than 5% then ranks are different, otherwise values have the
            same rank.

    Parameters
    ----------
    dbf : string
        Name of database.
    cls : string
        Name of classifier.

    Returns
    -------
    ress : 1D ndarray
        Array contains 13 elements with following meaning:
            0 - average BA for original database
            1 - average BA for PCA-K reduced database
            2 - average BA for PCA-BS reduced database
            3 - average BA for PCA-CN reduced database
            4-6 - average ranking for PCA-K, PCA-BS, and PCA-CN
            7-9 - t-test based ranking for PCA-K, PCA-BS, and PCA-CN
            10-12 - WSR test based ranking for PCA-K, PCA-BS, and PCA-CN

    '''
    # Create array for results
    ress = np.zeros((13))
    # Load data
    dat = np.loadtxt(f"results/Gen_{dbf}_{cls}.txt")
    # mean BA
    ress[0:4] = np.mean(dat, axis=0)
    # Data for t-test based ranking
    # who is the best?
    k = np.argmax(ress[1:4], keepdims=True)[0]
    kkk = np.argmin(ress[1:4], keepdims=True)[0]
    # Very special case when all three are equal
    if k == kkk:
        k = 0
        kk = 1
        kkk = 2
    else:
        # Define the second value
        if k == 0:
            if kkk == 1:
                kk = 2
            else:
                kk = 1
        elif k == 1:
            if kkk == 0:
                kk = 2
            else:
                kk = 0
        else:
            if kkk == 1:
                kk = 0
            else:
                kk = 1

    # T-test
    # the best
    ress[7 + k] = 0
    # Is the second SSD from the first
    if np.all(dat[:, k+1] == dat[:, kk+1]) or \
        (stats.ttest_rel(dat[:, k+1], dat[:, kk+1])[1] > 0.05):
        ress[7 + kk] = 0
        # Is the third SSD from the first
        if np.all(dat[:, k+1] == dat[:, kkk+1]) or \
            (stats.ttest_rel(dat[:, k+1], dat[:, kkk+1])[1] > 0.05):
            ress[7 + kkk] = 0
        else:
            ress[7 + kkk] = 1
    else:
        ress[7 + kk] = 1
        # Is the third SSD from the second
        if np.all(dat[:, kk+1] == dat[:, kkk+1]) or \
            (stats.ttest_rel(dat[:, kk+1], dat[:, kkk+1])[1] > 0.05):
            ress[7 + kkk] = 1
        else:
            ress[7 + kkk] = 2

    # Wilcoxon Signed-Rank test
    ress[10 + k] = 0
    # Is the second SSD from the first
    if np.all(dat[:, k+1] == dat[:, kk+1]) or \
        (stats.wilcoxon(dat[:, k+1], dat[:, kk+1])[1] > 0.05):
        ress[10 + kk] = 0
        # Is the third SSD from the first
        if np.all(dat[:, k+1] == dat[:, kkk+1]) or \
            (stats.wilcoxon(dat[:, k+1], dat[:, kkk+1])[1] > 0.05):
            ress[10 + kkk] = 0
        else:
            ress[10 + kkk] = 1
    else:
        ress[10 + kk] = 1
        # Is the third SSD from the second
        if np.all(dat[:, kk+1] == dat[:, kkk+1]) or \
            (stats.wilcoxon(dat[:, kk+1], dat[:, kkk+1])[1] > 0.05):
            ress[10 + kkk] = 1
        else:
            ress[10 + kkk] = 2

    # Final ranking
    # average rank
    ress[4:7] = stats.rankdata(1-ress[1:4])
    ress[7:10] = stats.rankdata(ress[7:10])
    ress[10:13] = stats.rankdata(ress[10:13])

    return ress

def aggregateBA(dbf):
    '''
    aggregateBA loads 100 files with results for folds for one database and
    created three files with information for all folds and one classifier is
    each.

    Parameters
    ----------
    dbf : string
        name of database to process.

    Returns
    -------
    None.

    '''
    print(dbf)
    # Create arrays for data
    knn = np.zeros((100,4))
    lr = np.zeros((100,4))
    lda = np.zeros((100,4))
    for fold in range(100):
        # Load file
        with open(f"results/Res_{dbf}_{fold:03d}.txt", encoding="utf-8") as file:
            content = file.readlines()
        knn[fold, 0] = float(content[1].split('\t')[2])
        knn[fold, 1] = float(content[4].split('\t')[2])
        knn[fold, 2] = float(content[7].split('\t')[2])
        knn[fold, 3] = float(content[10].split('\t')[2])
        lr[fold, 0] = float(content[2].split('\t')[2])
        lr[fold, 1] = float(content[5].split('\t')[2])
        lr[fold, 2] = float(content[8].split('\t')[2])
        lr[fold, 3] = float(content[11].split('\t')[2])
        lda[fold, 0] = float(content[3].split('\t')[2])
        lda[fold, 1] = float(content[6].split('\t')[2])
        lda[fold, 2] = float(content[9].split('\t')[2])
        lda[fold, 3] = float(content[12].split('\t')[2])
    # Write results
    np.savetxt(f"results/Gen_{dbf}_KNN.txt", knn, fmt='%8.6f')
    np.savetxt(f"results/Gen_{dbf}_LR.txt", lr, fmt='%8.6f')
    np.savetxt(f"results/Gen_{dbf}_LDA.txt", lda, fmt='%8.6f')


def recurrentDR(dbf):
    '''
    recurrentDR calculates eigenvalues for correlation matrix of database dbf

    Parameters
    ----------
    dbf : string
        name of database to calculate eigenvalues of correlation matrix.

    Returns
    -------
    fracs : 1D ndarray
        eigenvalues of correlation matrix.

    '''
    # Load database
    X, _ = readDatabase(dbf)
    # Calculate PCs
    pca = PCA(n_components=X.shape[1]).fit(X)
    fracs = pca.explained_variance_

    return fracs

def kNNFigures():
    '''
    kNNFigures depicts graphs of BA(k) for each database.
    Figures are saving to folder figures with name "bdName_KNN.png"

    Returns
    -------
    None.

    '''
    # Load data from kScores
    with open('kScores.txt', encoding="utf-8") as file:
        content = file.readlines()
    # Skip header
    content = content[1:]
    x = np.arange(1, 501)
    # load data for one database
    p = 0
    for line in content:
        tmp = line.split(",")
        opt = int(float(tmp[0]))
        n = len(tmp) - 2
        dat = np.zeros((n))
        for k in range(n):
            dat[k] = float(tmp[k + 1])
        # Form figure
        plt.figure()
        plt.plot(x[:n],dat,"k-")
        plt.title(f"Selection of optimal k for {databases[p]}")
        plt.xlabel("Number of nearest neighbours")
        plt.ylabel("Balanced accuracy")
        plt.vlines(opt, np.min(dat), np.max(dat), color='k')
        plt.text(opt+2, np.min(dat), f"Optimal k is {opt}")
        plt.savefig(f"figures/{databases[p]}_KNN.png")
        plt.close()
        p += 1

def oneDClass(scores, pure, score=balanced_accuracy_score, name=None):
    '''
    Function oneDClass applied classification with one input attribute by 
    searching the best threshold.

    Parameters
    ----------
    scores : 1D ndarray
        scores calculated for cases by some classifier..
    score : sklearn.metrics
        classification quality measure to optimise..
    pure : 1D ndarray
        corect class labels.
    name : list of string or None, optional
        DESCRIPTION. If name is not None then it should contain three elements, 
        used in titles on graphs:
            name(1) is name of attribute
            name(2) is name of the first class
            name(3) is name of the second class
        If name is None then graphs are not formed.
        The default is None.

    Returns
    -------
    bestTh : float
        Optimal threshold.
    Formed figure

    '''
    # Form list of possible thresholds
    th = np.unique(scores)
    th = (th[1:] + th[:-1]) / 2
    bestTh = th[0]
    bestSc = 0
    accs = np.zeros_like(th)
    p = 0
    for t in th:
        sc = score(pure, np.where(scores < t, 1, 0))
        accs[p] = sc
        p += 1
        if sc > bestSc:
            bestSc = sc
            bestTh = t
    
    # Now we are ready to form graph, if requested
    if not(name is None):
        # Define min and max to form bines
        mi = np.min(scores)
        ma = np.max(scores)
        edges = np.linspace(mi, ma, 21)
        
        ind = pure == 1
        # Draw histograms
        plt.figure();
        # histogram(x, edges, 'Normalization','probability');
        plt.hist(scores[ind], bins=edges, density=True, alpha=0.5, label=name[1])
        # hold on;
        # histogram(y, edges, 'Normalization','probability');
        plt.hist(scores[~ind], bins=edges, density=True, alpha=0.5, label=name[2])
        plt.title(name[0]);
        plt.xlabel("Value of " + name[0]);
        plt.ylabel('Fraction of cases');
        # Draw graph of errors
        sizes = plt.axis();
        plt.plot(th, accs * sizes[3], 'g', label="Quality measure");
        # Draw the best threshold
        plt.axvline(bestTh , color = 'k', label = 'Threshold')
        # plot([bestT, bestT], sizes(3:4), 'k', 'LineWidth', 2);
        plt.legend()

    return (bestTh, bestSc)

def fisherFigure():
    '''
    fisherFigure depicts graphs of optimal threshold selection for Fisher's 
    linear discriminant for each database.
    Figures are saving to folder figures with name "bdName_LDA.png"

    Returns
    -------
    None.

    '''
    # Database loop
    for dbf in databases:
        X, y = readDatabase(dbf)
        y = np.squeeze(y)
        # Form Fisher's direction
        # Fisher's discriminant
        fd = fisherDir(X, y)
        # Select threshold
        oneDClass(np.dot(X, fd), y, name = [f"LDA for {dbf}", 
                                            "Positive", "Negative"])
        plt.savefig(f"figures/{dbf}_LDA.png")
        plt.close()

    

# Define constants to use
databases = ["Banknote", "blood", "breastCancer", "climate",
             "Cryotherapy", "diabetic", "EggEyeState", "HTRU2",
             "Immunotherapy", "liver", "maledon",
             "minboone", "Musk", "Musk2", "plrx", "qsar",
             "skin", "sonar", "spect", "spectf",
             "telescope", "vertebral"]
dimNames = ["Full", "PCA-K", "PCA-BS", "PCA-CN"]
classifNames = ['KNN', 'LR', 'LDA']

# Score to use
scoreToUse = balanced_accuracy_score

# What is bitmask defined what to do in run
# Bit#, Value, Meaning
# 0      1      Convert databases from mat format to pkl
# 1      2      Form folders for testing protocol
# 2      4      TabErrorTest hypothesis that ID are independent
# 3      8      Search the best k for kNN
# 4     16      Process data
# 5     32      Recombine information for ranking
# 6     64      Calculation of ranks
# 7    128      Recurrent application of dimensionality reduction
# 8    256      Prepare figures for paper: k selection for kNN
# 9    512      Prepare figures for paper: threshold selection

# Specify what for calculation
what = 512

# Convert databases from mat format to pkl
if what & 1 != 0:
    for db in databases:
        databseConvert(db)

# Form folders for testing protocol
if what & 2 != 0:
    for db in databases:
        foldsForming(db)

# TabErrorTest hypothesis that ID are independent
if what & 4 != 0:
    res = dimCorrelations()

# Search the best k for kNN
if what & 8 != 0:
    # Form file for results
    with open("kScores.txt", "wb") as f:
        np.savetxt(f, np.arange(501), newline=", ")
    for db in databases:
        print(db)
        kks = kSearchKNN(db, scoreToUse)
        kks = np.concatenate((np.asarray([np.argmax(kks) + 1]), kks))
        with open("kScores.txt", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, kks, newline=", ")

# Process data
if what & 16 != 0:
    processData()

# 5    32       Recombine information for ranking
if what & 32 != 0:
    for db in databases:
        aggregateBA(db)

# 6     64      Calculation of ranks
if what & 64 != 0:
    # Create array for results
    n = len(databases)
    res = np.zeros((n, 13))
    for cl in classifNames:
        p=0
        for db in databases:
            print(db, cl)
            res[p, :] = ranksForOneDBAndClass(db, cl)

            # Next database
            p += 1

        # Save results for one classifier
        np.savetxt(f"Rank_{cl}.txt", res, fmt='%8.6f')

# 7    128      Recurrent application of dimensionality reduction
if what & 128 != 0:
    res = recurrentDR("Musk2")
    # Vector of eigenvalues res was copied to excel.

# 8    256      Prepare figures for paper: k selection for kNN
if what & 256 != 0:
    kNNFigures()

# 9    512      Prepare figures for paper: threshold selection
if what & 512 != 0:
    fisherFigure()
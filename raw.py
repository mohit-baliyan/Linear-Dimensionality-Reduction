# # files = os.listdir( './Databases/' )
# #
# # Databases = []
# #
# # Thresholds = []
# #
# #
# # for file in files :
# #
# #
# #     # read one database at a time
# #
# #     data = loadmat( './Databases/' + file )
# #
# #     X = data['X']
# #
# #     # standarise data
# #
# #     X = stats.zscore( X )
# #
# #
# #
# #     y = data['Y']
# #
# #     ( m , n ) = X.shape
# #
# #
# #     try :
# #
# #         model = LogitRegression( learning_rate = 0.01, iterations = 500 )
# #
# #         model.fit( X, y )
# #
# #         z = model.predict( X )
# #
# #         bestthres = threshold_selection( z, y )
# #
# #         Databases.append( file )
# #
# #         Thresholds.append( bestthres )
# #
# #     except :
# #
# #
# #         continue
# #
# #
# # # In[ ]:
# #
# #
# # Thres = { 'Databases' : Databases, 'Thresholds' : Thresholds }
# #
# # Thres = pd.DataFrame( Thres )
# #
# # Thres
# #
# #
# # # In[ ]:
# #
# #
# # Thres.to_csv( 'Thres.csv' )

# # # # Modelling and Balanced Accuracy calculation 10 times
# #
# # # In[39]:
# #
# #
# # # return the average of balanced accuracy after running 10 times with 10 fold stratified cross-validation
# #
# # def magic(X, y, model):
# #     # outer loop to calculate the balanced accuracy 10 times
# #
# #     balanced_accuracies = []
# #
# #     for i in range(0, 10):
# #
# #         # shuffle X, y before Splitting
# #
# #         shuffle(X, y)
# #
# #         skfold = StratifiedKFold(n_splits=10, shuffle=True)
# #
# #         balanced_accuracy_K_folds = []
# #
# #         # inner loop for 10 fold stratified cross validation
# #
# #         for train_index, test_index in skfold.split(X, y):
# #             X_train, X_test = X[train_index], X[test_index]
# #
# #             y_train, y_test = y[train_index], y[test_index]
# #
# #             model.fit(X_train, y_train)
# #
# #             balanced_accuracy_K_folds.append(balanced_accuracy_score(y_test, model.predict(X_test)))
# #
# #         balanced_accuracies.append(np.mean(balanced_accuracy_K_folds))
# #
# #     return np.mean(balanced_accuracies)
# #
# #
# # # # KNN Results
# #
# # # In[40]:
# #
# #
# # Thres = pd.read_csv('Thres.csv')
# #
# # Component_Selection = pd.read_csv('Component_Selection.csv')
# #
# # Databases = Thres["Databases"]
# #
# # # In[41]:
# #
# #
# # Databases = Databases[:10]
# #
# # # In[42]:
# #
# #
# # BA_K = []
# #
# # BA_BS = []
# #
# # BA_CN = []
# #
# # for file in Databases:
# #     # Get X and Y
# #
# #     data = loadmat('Databases/' + file)
# #
# #     X = data['X']
# #
# #     # standarise data
# #
# #     X = stats.zscore(X)
# #
# #     y = data['Y']
# #
# #     # select row from Component_Selection dataframe
# #
# #     x = Component_Selection.loc[Component_Selection['Databases'] == file]
# #
# #     # For Kaiser-rule
# #
# #     no_of_components_kaiser_rule = x["PCA-K"].values[0]
# #
# #     pca = PCA()
# #
# #     X_kaiser_rule = pca.transformation(X, no_of_components_kaiser_rule)
# #
# #     knn = K_Nearest_Neighbors_Classifier(K=3)
# #
# #     BA_K.append(magic(X_kaiser_rule, y, knn))
# #
# #     # For Broken Stick
# #
# #     no_of_components_broken_stick = x["PCA-BS"].values[0]
# #
# #     pca = PCA()
# #
# #     X_broken_stick = pca.transformation(X, no_of_components_broken_stick)
# #
# #     knn = K_Nearest_Neighbors_Classifier(K=3)
# #
# #     BA_BS.append(magic(X_broken_stick, y, knn))
# #
# #     # For conditional number
# #
# #     no_of_components_conditonal_number = x["PCA-CN"].values[0]
# #
# #     pca = PCA()
# #
# #     X_conditional_number = pca.transformation(X, no_of_components_conditonal_number)
# #
# #     knn = K_Nearest_Neighbors_Classifier(K=3)
# #
# #     BA_CN.append(magic(X_conditional_number, y, knn))
# #
# # # In[43]:
# #
# # Databases = Databases.to_list()
# #
# # # In[44]:
# #
# #
# # KNN_Results = {'Databases': Databases, 'BA_K': BA_K, 'BA_BS': BA_BS, 'BA_CN': BA_CN}
# #
# # KNN_Results = pd.DataFrame(KNN_Results)
# #
# # KNN_Results.to_csv('KNN_Results.csv')
# #
# # # In[45]:
# #
# #
# # KNN_Results
# #
# # # # Logistic Regression Results
# #
# # # In[46]:
# #
# #
# # BA_K = []
# #
# # BA_BS = []
# #
# # BA_CN = []
# #
# # for file in Databases:
# #
# #     # Get X and Y
# #
# #     data = loadmat('Databases/' + file)
# #
# #     X = data['X']
# #
# #     # standarise data
# #
# #     X = stats.zscore(X)
# #
# #     y = data['Y']
# #
# #     # select row from Component_Selection dataframe
# #
# #     x = Component_Selection.loc[Component_Selection['Databases'] == file]
# #
# #     # select row from Component_Selection dataframe
# #
# #     z = Thres.loc[Thres["Databases"] == file]
# #
# #     best_thres = z["Thresholds"].values[0]
# #
# #     if (math.isnan(best_thres)):
# #         best_thres = 0.5
# #
# #     # For Kaiser-rule
# #
# #     no_of_components_kaiser_rule = x["PCA-K"].values[0]
# #
# #     pca = PCA()
# #
# #     X_kaiser_rule = pca.transformation(X, no_of_components_kaiser_rule)
# #
# #     model = LogitRegression(0.01, 500, best_thres)
# #
# #     BA_K.append(magic(X_kaiser_rule, y, model))
# #
# #     # For Broken Stick
# #
# #     no_of_components_broken_stick = x["PCA-BS"].values[0]
# #
# #     pca = PCA()
# #
# #     X_broken_stick = pca.transformation(X, no_of_components_broken_stick)
# #
# #     model = LogitRegression(0.01, 500, best_thres)
# #
# #     BA_BS.append(magic(X_broken_stick, y, model))
# #
# #     # For conditional number
# #
# #     no_of_components_conditonal_number = x["PCA-CN"].values[0]
# #
# #     pca = PCA()
# #
# #     X_conditional_number = pca.transformation(X, no_of_components_conditonal_number)
# #
# #     model = LogitRegression(0.01, 500, best_thres)
# #
# #     BA_CN.append(magic(X_conditional_number, y, model))
# #
# # # In[47]:
# #
# # Logistic_Results = {'Databases': Databases, 'BA_K': BA_K, 'BA_BS': BA_BS, 'BA_CN': BA_CN}
# #
# # Logistic_Results = pd.DataFrame(Logistic_Results)
# #
# # # Logistic_Results.to_csv(Logistic_Results.csv')
# #
# # # In[48]:
# #
# #
# # Logistic_Results
# #
# # # In[50]:
# #
# #
# # # while( True ) :
# #
# # #     code()
# #
# # #     eat()
# #
# # #     sleep()
# #
# #
# # # # Fisher LDA Results
# #
# #
# #
# #
# # # # return the average of balanced accuracy after running 10 times with 10-fold stratified cross-validation
# # # def calculate_accuracy(X, y, model, database):
# # #
# # #     # create folds and write to HDD if folds does not exist
# # #     if not os.path.isdir('folds_' + database):
# # #         save_folds(X, y, database)
# # #
# # #     # outer loop to calculate the balanced accuracy 10 times
# # #     balanced_accuracies = []
# # #
# # #     i = 0
# # #     for i in range(0, 10):
# # #         balanced_accuracy_10_folds = []
# # #
# # #         j = 1
# # #         # inner loop for 10-fold stratified cross validation
# # #         for j in range(1, 11):
# # #             # read folds from csv and convert into numpy array
# # #             df_train = pd.read_csv('folds_' + database + '/train_fold_' + str(j) + '.csv', index_col=0)
# # #             df_test = pd.read_csv('folds_' + database + '/val_fold_' + str(j) + '.csv', index_col=0)
# # #             df_train = df_train.values
# # #             df_test = df_test.values
# # #
# # #             # slicing into X_train, y_train, X_test and y_test
# # #             X_train = df_train[:, :-1]
# # #             y_train = df_train[:, -1]
# # #             X_test = df_test[:, :-1]
# # #             y_test = df_test[:, -1]
# # #
# # #             # train model and predict X_test
# # #             model.fit(X_train, y_train)
# # #             balanced_accuracy_10_folds.append(balanced_accuracy_score(y_test, model.predict(X_test)))
# # #
# # #         balanced_accuracies.append(np.mean(balanced_accuracy_10_folds))
# # #
# # #     return np.mean(balanced_accuracies)
# #
#         # test_index = np.loadtext('folds_' + database + '/test_fold_' + str(fold_no) + '.txt')
# #
# #
# #
#
#
#
#
#
#
# # import pandas as pd
# # from knn import KNN
# # from scipy.io import loadmat
# # from scipy.stats import stats
# # from pca import PCA, SelectionMethods
# # from create_folds import calculate_accuracy
# #
# #
# # def main():
# #
# #     # load dataset
# #     data = loadmat('./Databases/Musk.mat')
# #     X = data['X']
# #     y = data['Y']
# #
# #     # standardise data
# #     X = stats.zscore(X)
# #
# #     # apply pca with conditional number
# #     pca = PCA()
# #     method = SelectionMethods(X)
# #     conditional_comps = method.conditional_number()
# #     X_transform = pca.transformation(X, conditional_comps)
# #
# #     # knn modelling and save results for different values of k
# #     k = []
# #     accuracy = []
# #     for i in range(1, 30):
# #         knn = KNN(i)
# #         k.append(i)
# #         accuracy.append(calculate_accuracy(X_transform, y, knn, 'MuskCN'))
# #
# #     # create dictionary, then convert into pandas dataframe to further save as a csv file
# #     dictionary = {'K': k, 'Accuracy': accuracy}
# #     df = pd.DataFrame(dictionary)
# #     df.to_csv("neighbors.csv")
# #
# #
# # if __name__ == "__main__":
# #     main()
#
#
#
# from pca import PCA
# from knn import KNN
# import pandas as pd
# from scipy.io import loadmat
# from scipy.stats import stats
# from create_folds import calculate_accuracy
#
#
# def main():
#
#     # read dimensions
#     df = pd.read_csv('dimensions.csv')
#
#     # load database
#     database = 'telescope.mat'
#     data = loadmat('./Databases/' + database)
#     X = data['X']
#     y = data['Y']
#
#     # standardise data
#     X = stats.zscore(X)
#
#     # initialize KNN model
#     knn = KNN(3)
#
#     # without transformation
#     accuracy_original = calculate_accuracy(X, y, knn, database[:-4]+'O')
#
#     # initialize pca
#     pca = PCA()
#
#     # after kaiser rule
#     accuracy_kaiser = calculate_accuracy(pca.transformation(X, df[df['Databases'] == database]['PCA-K'].item()), y,
#                                          knn, database[:-4]+'K')
#
#     # after broken stick
#     accuracy_bs = calculate_accuracy(pca.transformation(X, df[df['Databases'] == database]['PCA-BS'].item()), y,
#                                      knn, database[:-4]+'BS')
#
#     # after conditional number
#     accuracy_cn = calculate_accuracy(pca.transformation(X, df[df['Databases'] == database]['PCA-CN'].item()), y, knn,
#                                      database[:-4]+'CN')
#
#     print(database, round(accuracy_original, 2), round(accuracy_kaiser, 2), round(accuracy_bs, 2),
#           round(accuracy_cn, 2))
#
#
# if __name__ == "__main__":
#     main()





















import csv
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings('ignore')

'''
    Since we are looking at dips in the data, we should remove upper outliers.
    The function is taken from here:
    https://www.kaggle.com/muonneutrino/exoplanet-data-visualization-and-exploration
'''
def reduce_upper_outliers(df,reduce = 0.01, half_width=4):    
    length = len(df.iloc[0,:])
    remove = int(length*reduce)
    for i in df.index.values:
        values = df.loc[i,:]
        sorted_values = values.sort_values(ascending = False)
        for j in range(remove):
            idx = sorted_values.index[j]
            new_val = 0
            count = 0
            idx_num = int(idx[5:])
            for k in range(2*half_width+1):
                idx2 = idx_num + k - half_width
                if idx2 <1 or idx2 >= length or idx_num == idx2:
                    continue
                new_val += values['FLUX.'+str(idx2)]
                count += 1
            new_val /= count
            if new_val < values[idx]:
                df.at[i,idx]=new_val
    return df


# Tweak parameters here, like n_splits=10,11,...
def model_evaluator(X, y, model, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits)
    
    bootstrapped_accuracies = list()
    bootstrapped_precisions = list()
    bootstrapped_recalls    = list()
    bootstrapped_f1s        = list()
    
    SMOTE_accuracies = list()
    SMOTE_precisions = list()
    SMOTE_recalls    = list()
    SMOTE_f1s        = list()

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        df_train    = X_train.join(y_train)
        df_planet   = df_train[df_train.LABEL == 2].reset_index(drop=True)
        df_noplanet = df_train[df_train.LABEL == 1].reset_index(drop=True)
        df_boot     = df_noplanet
                        
        index = np.arange(0, df_planet.shape[0])
        temp_index = np.random.choice(index, size=df_noplanet.shape[0])
        df_boot = df_boot.append(df_planet.iloc[temp_index])
        
        df_boot = df_boot.reset_index(drop=True)
        X_train_boot = df_boot.drop('LABEL', axis=1)
        y_train_boot = df_boot.LABEL
                    
        est_boot = model.fit(X_train_boot, y_train_boot)
        y_test_pred = est_boot.predict(X_test)
        
        bootstrapped_accuracies.append(accuracy_score(y_test, y_test_pred))
        bootstrapped_precisions.append(precision_score(y_test, y_test_pred, pos_label=2))
        bootstrapped_recalls.append(recall_score(y_test, y_test_pred, pos_label=2))
        bootstrapped_f1s.append(f1_score(y_test, y_test_pred, pos_label=2))

        # Try with 'auto' and 1.0
        sm = SMOTE(ratio = 1.0)
        X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)
                    
        est_sm = model.fit(X_train_sm, y_train_sm)
        y_test_pred = est_sm.predict(X_test)
        
        SMOTE_accuracies.append(accuracy_score(y_test, y_test_pred))
        SMOTE_precisions.append(precision_score(y_test, y_test_pred, pos_label=2))
        SMOTE_recalls.append(recall_score(y_test, y_test_pred, pos_label=2))
        SMOTE_f1s.append(f1_score(y_test, y_test_pred, pos_label=2))
        print ("lol")
        
    print('\t\t\t Bootstrapped \t SMOTE')
    print("Average Accuracy:\t", "{:0.10f}".format(np.mean(bootstrapped_accuracies)),
          '\t', "{:0.10f}".format(np.mean(SMOTE_accuracies)))
    print("Average Precision:\t", "{:0.10f}".format(np.mean(bootstrapped_precisions)),
          '\t', "{:0.10f}".format(np.mean(SMOTE_precisions)))
    print("Average Recall:\t\t", "{:0.10f}".format(np.mean(bootstrapped_recalls)),
          '\t', "{:0.10f}".format(np.mean(SMOTE_recalls)))
    print("Average F1:\t\t", "{:0.10f}".format(np.mean(bootstrapped_f1s)),
          '\t', "{:0.10f}".format(np.mean(SMOTE_f1s)))


def spectrum_getter(X):
    Spectrum = scipy.fft(X, n=X.size)
    return np.abs(Spectrum)

def detrender_normalizer(X):
    flux1 = X
    flux2 = ndimage.filters.gaussian_filter(flux1, sigma=10)
    flux3 = flux1 - flux2
    flux3normalized = (flux3-np.mean(flux3)) / (np.max(flux3)-np.min(flux3))
    return flux3normalized


# Raw data
extrain = pd.read_csv('Data/ExoTrain.csv')
extrain, extest = np.split(extrain, [int(0.80*len(extrain))])

# Detrending data to make box-fitting possible.
extrain.iloc[:,1:] = extrain.iloc[:,1:].apply(detrender_normalizer,axis=1)
extest.iloc[:,1:] = extest.iloc[:,1:].apply(detrender_normalizer,axis=1)

# Removing upper outliers, because we don't need them.
extrain.iloc[:,1:] = reduce_upper_outliers(extrain.iloc[:,1:])
extest.iloc[:,1:] = reduce_upper_outliers(extest.iloc[:,1:])

# These X and y are input and output of the training data which is processed till now.
X = extrain.drop('LABEL', axis=1)
y = extrain.LABEL
 

# Applying FFT to get a trend in the frequency distribution in the data as well.
X_train = extrain.drop('LABEL', axis=1)
y_train = extrain.LABEL

X_test = extest.drop('LABEL', axis=1)
y_test = extest.LABEL

new_X_train = X_train.apply(spectrum_getter,axis=1)
new_X_test = X_test.apply(spectrum_getter,axis=1)

X = new_X_train
y = y_train

X_final_test = new_X_test
y_final_test = y_test

X = X.iloc[:,:(X.shape[1]//2)]
X_final_test = X_final_test.iloc[:,:(X_final_test.shape[1]//2)]

X_columns = np.arange(len(X.columns))
X_columns = X_columns * (1.0/(36*60)) # sampling frequency of our data
X.columns = X_columns
X_final_test.columns = X_columns


# Put this function at any stage above to test results at various stages.
# We can make a final classifier with different weightage to classification using these different stages.
model_evaluator(X, y, KNeighborsClassifier(n_jobs=-1))
# Try other classifiers too, like some written below

# model_evaluator(X, y, LinearSVC())
# model_evaluator(X, y, LogisticRegression(n_jobs=-1))

# Further test by normalization
X = pd.DataFrame(normalize(X))
X_final_test = pd.DataFrame(normalize(X_final_test))

# model_evaluator(X, y, KNeighborsClassifier(n_jobs=-1))
# Try other classifiers too, like some written below

# model_evaluator(X, y, LinearSVC())
# model_evaluator(X, y, LogisticRegression(n_jobs=-1))

# Test these and put the results in the report. Write what all we tried.




# Everything below this is for further paramter optimization using grid and randomized search, and then getting the final results of the test.
# Do this after you've done the testing an found an optimum classifier.
# Write everything in the report, from what you observe, what I observed, what trends, what methods everything.
# With some formulas and everything.




# Looks like we found our model to optimize! We can now look closely at Linear Support Vector Classification and find most optimal parameters for it via randomized and grid searches. Perhaps, we can improve the recall rate.  
# # ## Linear SVC Optimization through Randomized and Grid Searches

# # Let's see if the Randomized Search works with our data without taking it through SMOTE process. For that I will be setting `class_weight` parameter in `LinearSVC` classifier to `'balanced'`. Also, I am unsure how to tell Randomized Search that I am looking for LABEL = 2, so to avoid any confusion, I will convert all 1s to 0s and all 2s to 1s:

# # In[ ]:


# y_new = y - 1


# # In[ ]:


# from sklearn.model_selection import RandomizedSearchCV

# # develop your "tuned parameters"

# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# print(__doc__)

# # Split the dataset in two equal parts
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_new, test_size=0.5, stratify=y)

# # Set the parameters by cross-validation
# tuned_parameters = [{'penalty': ['l2'], #'l1'],
#               'loss': ['hinge'],
#               'dual': [True],
#               'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
#               'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
#               'fit_intercept': [True, False],
#               'intercept_scaling': [0.01, 0.1, 1, 10, 100],
#               'class_weight': ['balanced'],
#               'verbose': [0],
#               'random_state': [None],
#               'max_iter': [10, 100, 1000, 10000, 100000]},
#                     {'penalty': ['l2'], #'l1'],
#               'loss': ['squared_hinge'],
#               'dual': [True, False],
#               'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
#               'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
#               'fit_intercept': [True, False],
#               'intercept_scaling': [0.01, 0.1, 1, 10, 100],
#               'class_weight': ['balanced'],
#               'verbose': [0],
#               'random_state': [None],
#               'max_iter': [10, 100, 1000, 10000, 100000]},
#                    {'penalty': ['l1'],
#               'loss': ['squared_hinge'], #'hinge'],
#               'dual': [False],
#               'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
#               'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
#               'fit_intercept': [True, False],
#               'intercept_scaling': [0.01, 0.1, 1, 10, 100],
#               'class_weight': ['balanced'],
#               'verbose': [0],
#               'random_state': [None],
#               'max_iter': [10, 100, 1000, 10000, 100000]}]

# model_scores = ['precision', 'recall', 'f1']

# for model_score in model_scores:
#     print("# Tuning hyper-parameters for %s" % model_score)
#     print()
#     for tuned_parameter in tuned_parameters:
#         clf = RandomizedSearchCV(LinearSVC(), tuned_parameter, cv=3, scoring=model_score, n_jobs=-1)
#         clf.fit(X_train, y_train)

#         print("Best parameters set found on development set:")
#         print()
#         print(clf.best_estimator_)
#         print()

#         print("Detailed classification report:")
#         print()
#         print("The model is trained on the full development set.")
#         print("The scores are computed on the full evaluation set.")
#         print()
#         y_true, y_pred = y_test, clf.predict(X_test)
#         print(classification_report(y_true, y_pred))
#         print()


# # Let's see if we can do better with SMOTE data balancing. This time `class_weight` is going to be `None` since the data will be balanced through synthetic data generation:

# # In[ ]:


# def SMOTE_synthesizer(X, y):
#         sm = SMOTE(ratio = 1.0)
#         X, y = sm.fit_sample(X, y)
#         return X, y


# # In[ ]:


# from sklearn.model_selection import RandomizedSearchCV

# # develop your "tuned parameters"

# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# print(__doc__)

# # Split the dataset in two equal parts
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_new, test_size=0.5, stratify=y)

# X_train, y_train = SMOTE_synthesizer(X_train, y_train)

# # Set the parameters by cross-validation
# tuned_parameters = [{'penalty': ['l2'], #'l1'],
#               'loss': ['hinge'],
#               'dual': [True],
#               'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
#               'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
#               'fit_intercept': [True, False],
#               'intercept_scaling': [0.01, 0.1, 1, 10, 100],
#               'class_weight': [None],
#               'verbose': [0],
#               'random_state': [None],
#               'max_iter': [10, 100, 1000, 10000, 100000]},
#                     {'penalty': ['l2'], #'l1'],
#               'loss': ['squared_hinge'],
#               'dual': [True, False],
#               'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
#               'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
#               'fit_intercept': [True, False],
#               'intercept_scaling': [0.01, 0.1, 1, 10, 100],
#               'class_weight': [None],
#               'verbose': [0],
#               'random_state': [None],
#               'max_iter': [10, 100, 1000, 10000, 100000]},
#                    {'penalty': ['l1'],
#               'loss': ['squared_hinge'], #'hinge'],
#               'dual': [False],
#               'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
#               'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
#               'fit_intercept': [True, False],
#               'intercept_scaling': [0.01, 0.1, 1, 10, 100],
#               'class_weight': [None],
#               'verbose': [0],
#               'random_state': [None],
#               'max_iter': [10, 100, 1000, 10000, 100000]}]

# model_scores = ['precision', 'recall', 'f1']

# for model_score in model_scores:
#     print("# Tuning hyper-parameters for %s" % model_score)
#     print()
#     for tuned_parameter in tuned_parameters:
#         clf = RandomizedSearchCV(LinearSVC(), tuned_parameter, cv=3, scoring=model_score, n_jobs=-1)
#         clf.fit(X_train, y_train)

#         print("Best parameters set found on development set:")
#         print()
#         print(clf.best_estimator_)
#         print()

#         print("Detailed classification report:")
#         print()
#         print("The model is trained on the full development set.")
#         print("The scores are computed on the full evaluation set.")
#         print()
#         y_true, y_pred = y_test, clf.predict(X_test)
#         print(classification_report(y_true, y_pred))
#         print()


# # So at the end of the day, it still looks like training on synthetic data works better than balancing. For models trained with SMOTE data, my best `precision`, `recall`, and `f1` scores were `1.00`, `0.47`, and `0.64`, respectively. For models that try to balance training data, my best `precision`, `recall`, and `f1` scores were `1.00`, `0.37`, and `0.54`, respectively. The best scores for SMOTE trained models were achieved with the following parameters:
# # ```python
# # LinearSVC(
# #     C=100.0,
# #     class_weight=None,
# #     dual=True,
# #     fit_intercept=True,
# #     intercept_scaling=10,
# #     loss='hinge',
# #     max_iter=100000,
# #     multi_class='ovr',
# #     penalty='l2',
# #     random_state=None,
# #     tol=0.0001,
# #     verbose=0)```
# # Therefore, we shall now proceed with the Grid Search to zoom in on possible settings around these parameters:

# # In[ ]:


# from sklearn.grid_search import GridSearchCV

# # develop your "tuned parameters"

# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report

# print(__doc__)

# # Split the dataset in two equal parts
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_new, test_size=0.5, stratify=y)

# X_train, y_train = SMOTE_synthesizer(X_train, y_train)

# # Set the parameters by cross-validation
# tuned_parameters = {'penalty': ['l2'],
#               'loss': ['hinge'],
#               'dual': [True],
#               'tol': np.arange(0.00008, .00014, 0.00002),
#               'C': list(range(100, 220, 20)),
#               'fit_intercept': [True],
#               'intercept_scaling': np.arange(10, 130, 30),
#               'class_weight': [None],
#               'verbose': [0],
#               'random_state': [None]}
# #               'max_iter': np.arange(10000, 130000, 30000)}

# # LinearSVC(
# #     C=100.0,
# #     class_weight=None,
# #     dual=True,
# #     fit_intercept=True,
# #     intercept_scaling=10,
# #     loss='hinge',
# #     max_iter=100000,
# #     multi_class='ovr',
# #     penalty='l2',
# #     random_state=None,
# #     tol=0.0001,
# #     verbose=0)

# model_scores = ['precision', 'recall', 'f1']

# for model_score in model_scores:
#     print("# Tuning hyper-parameters for %s" % model_score)
#     print()

#     clf = GridSearchCV(LinearSVC(), tuned_parameters, cv=3, scoring=model_score, n_jobs=-1)
#     clf.fit(X_train, y_train)

#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_estimator_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     for params, mean_score, scores in clf.grid_scores_:
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean_score, scores.std() / 2, params))
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()


# # Unfortunately, none of the parameter searches yielded results better than our initial default LinearSVC() model. Therefore, we shall proceed and create our final model uising default parameters of LinearSVC():

# # ## Final Model

# # In[ ]:


# X_sm, y_sm = SMOTE_synthesizer(X, y)


# # In[ ]:


# final_model = LinearSVC()
# final_model.fit(X_sm, y_sm)


# # In[ ]:


# y_pred = final_model.predict(X_final_test)


# # In[ ]:


# from sklearn.metrics import classification_report
# print(classification_report(y_final_test, y_pred))
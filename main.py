
# coding: utf-8

# # Hunting for Exoplanets   
# ## Initial Data Processing

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy import signal
from  scipy import ndimage
from IPython import *
import matplotlib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

import jupyternotify
ip = get_ipython()
ip.register_magics(jupyternotify.JupyterNotifyMagics)


# In[ ]:


extrain = pd.read_csv('Data/exoTrain.csv')
extest = pd.read_csv('Data/exoTest.csv')


# What does the data look like?

# In[ ]:


extrain.head()


# Each row has a label and tnen time series of light flux.  
# 
# According to [NASA](https://keplerscience.arc.nasa.gov/k2-observing.html),
# > K2 observations entail a series of sequential observing "Campaigns" of fields distributed around the ecliptic plane. Each ecliptic Campaign is limited by Sun angle constraints to a duration of approximately 80 days as illustrated in the image below (Howell et al. 2014). Therefore, four to five K2 Campaigns can be performed during each 372-day orbit of the spacecraft.  
# ![kepler campaigns](https://www.nasa.gov/sites/default/files/k2_explained_25nov_story.jpg)
# 
# Therefore, observations are (80 days) / (3197 columns) or approximately 36 minutes apart, i.e. the sampling frequency is 1 / (36 minutes * 60 seconds in a minute) or 0.00046 Hz.

# What does the flux look like for a star without confirmed exoplanets vs. a star with confirmed exoplanets?

# In[ ]:


# Obtaining flux for several stars without exoplanets from the train data:
for i in [0, 999, 1999, 2999, 3999, 4999]:
    flux = extrain[extrain.LABEL == 1].drop('LABEL', axis=1).iloc[i,:]
    time = np.arange(len(flux)) * (36.0/60.0) # time in units of hours
    plt.figure(figsize=(15,5))
    plt.title('Flux of star {} with no confirmed exoplanets'.format(i+1))
    plt.ylabel('Flux, e-/s')
    plt.xlabel('Time, hours')
    plt.plot(time, flux)


# In[ ]:


# Obtaining flux for several stars without exoplanets from the train data:
for i in [0, 9, 14, 19, 24, 29]:
    flux = extrain[extrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
    time = np.arange(len(flux)) * (36.0/60.0) # time in units of hours
    plt.figure(figsize=(15,5))
    plt.title('Flux of star {} with confirmed exoplanets'.format(i+1))
    plt.ylabel('Flux, e-/s')
    plt.xlabel('Time, hours')
    plt.plot(time, flux)


# There are a lot of different shapes and magnitudes in the data. Therefore, the data needs to be transformed to one single standard in order for us to produce a better model:  
# 
# ### 1. Detrend data  
# We have noticed many different shapes in the signal. Removing any trends can go a long way in further processing:

# In[ ]:


i = 13
flux1 = extrain[extrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
time = np.arange(len(flux1)) * (36.0/60.0) # time in units of hours
plt.figure(figsize=(15,5))
plt.title('Original flux of star {} with confirmed exoplanets'.format(i+1))
plt.ylabel('Flux, e-/s')
plt.xlabel('Time, hours')
plt.plot(time, flux1)


# In order to detrend, we need to find a general shape of the signal which we can then subtract it form the original signal to get its relatively flat representation. I decided to use Gaussian smoothing:

# In[ ]:


i = 13
flux2 = ndimage.filters.gaussian_filter(flux1, sigma=10)
time = np.arange(len(flux2)) * (36.0/60.0) # time in units of hours
plt.figure(figsize=(15,5))
plt.title('Smoothed flux of star {} with confirmed exoplanets'.format(i+1))
plt.ylabel('Flux, e-/s')
plt.xlabel('Time, hours')
plt.plot(time, flux2)


# Detrend the signal:

# In[ ]:


i = 13
flux3 = flux1 - flux2
time = np.arange(len(flux3)) * (36.0/60.0) # time in units of hours
plt.figure(figsize=(15,5))
plt.title('Detrended flux of star {} with confirmed exoplanets'.format(i+1))
plt.ylabel('Flux, e-/s')
plt.xlabel('Time, hours')
plt.plot(time, flux3)


# ### 2. Normalize the detrended signal

# In[ ]:


i = 13
flux3normalized = (flux3-np.mean(flux3))/(np.max(flux3)-np.min(flux3))
time = np.arange(len(flux3normalized)) * (36.0/60.0) # time in units of hours
plt.figure(figsize=(15,5))
plt.title('Normalized detrended flux of star {} with confirmed exoplanets'.format(i+1))
plt.ylabel('Normalized flux')
plt.xlabel('Time, hours')
plt.plot(time, flux3normalized)


# Let's apply this process to the entirety of the data:

# In[ ]:


def detrender_normalizer(X):
    flux1 = X
    flux2 = ndimage.filters.gaussian_filter(flux1, sigma=10)
    flux3 = flux1 - flux2
    flux3normalized = (flux3-np.mean(flux3)) / (np.max(flux3)-np.min(flux3))
    return flux3normalized


# In[ ]:


extrain.iloc[:,1:] = extrain.iloc[:,1:].apply(detrender_normalizer,axis=1)
extest.iloc[:,1:] = extest.iloc[:,1:].apply(detrender_normalizer,axis=1)


# In[ ]:


extrain.head()


# In[ ]:


i = 13
flux1 = extrain[extrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
flux1 = flux1.reset_index(drop=True)
time = np.arange(len(flux1)) * (36.0/60.0) # time in units of hours
plt.figure(figsize=(15,5))
plt.title('Processed flux of star {} with confirmed exoplanets'.format(i+1))
plt.ylabel('Flux, e-/s')
plt.xlabel('Time, hours')
plt.plot(time, flux1)


# Since we are looking for dips in flux when exoplanets pass between the telescope and the star, we should remove any upper outliers:

# In[ ]:


def reduce_upper_outliers(df,reduce = 0.01, half_width=4):
    '''
    Since we are looking at dips in the data, we should remove upper outliers.
    The function is taken from here:
    https://www.kaggle.com/muonneutrino/exoplanet-data-visualization-and-exploration
    '''
    length = len(df.iloc[0,:])
    remove = int(length*reduce)
    for i in df.index.values:
        values = df.loc[i,:]
        sorted_values = values.sort_values(ascending = False)
       # print(sorted_values[:30])
        for j in range(remove):
            idx = sorted_values.index[j]
            #print(idx)
            new_val = 0
            count = 0
            idx_num = int(idx[5:])
            #print(idx,idx_num)
            for k in range(2*half_width+1):
                idx2 = idx_num + k - half_width
                if idx2 <1 or idx2 >= length or idx_num == idx2:
                    continue
                new_val += values['FLUX.'+str(idx2)] # corrected from 'FLUX-' to 'FLUX.'
                
                count += 1
            new_val /= count # count will always be positive here
            #print(new_val)
            if new_val < values[idx]: # just in case there's a few persistently high adjacent values
                df.set_value(i,idx,new_val)
        
            
    return df


# In[ ]:


extrain.iloc[:,1:] = reduce_upper_outliers(extrain.iloc[:,1:])
extest.iloc[:,1:] = reduce_upper_outliers(extest.iloc[:,1:])


# In[ ]:


i = 13
flux1 = extrain[extrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
flux1 = flux1.reset_index(drop=True)
time = np.arange(len(flux1)) * (36.0/60.0) # time in units of hours
plt.figure(figsize=(15,5))
plt.title('Processed flux of star {} with confirmed exoplanets (removed upper outliers)'.format(i+1))
plt.ylabel('Normalized flux')
plt.xlabel('Time, hours')
plt.plot(time, flux1)


# ## Baseline Model
# Before we try to generate any features from the data, let's see if we can get any good models just by feeding them raw data

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold


# There is a huge imbalance in classes with LABEL = 1 (no exoplanets) being the dominant class:

# In[ ]:


extrain.LABEL.value_counts()


# In[ ]:


extest.LABEL.value_counts()


# That means that I will have to perform test-train splits with stratification.

# Therefore, we either need to bootstrap the data to get more LABEL = 2 observations, or we can synthetically generate new data using SMOTE approach, see more information [here](https://beckernick.github.io/oversampling-modeling/). Bootstrapping/using SMOTE with impbalanced data should proceed in the following order:
# 1. Perform stratified test-train separation on the data
# 2. Boostrap/use SMOTE on the train data, leave test data untouched
# 3. Perform model fit on the bootstrapped train data
# 4. Assess you model's performance using untouched test data  
# The code below also performs K-fold validation with the default of 5 folds:

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


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
    
        sm = SMOTE(ratio = 1.0)
        X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)
                    
        est_sm = model.fit(X_train_sm, y_train_sm)
        y_test_pred = est_sm.predict(X_test)
        
        SMOTE_accuracies.append(accuracy_score(y_test, y_test_pred))
        SMOTE_precisions.append(precision_score(y_test, y_test_pred, pos_label=2))
        SMOTE_recalls.append(recall_score(y_test, y_test_pred, pos_label=2))
        SMOTE_f1s.append(f1_score(y_test, y_test_pred, pos_label=2))
        
    print('\t\t\t Bootstrapped \t SMOTE')
    print("Average Accuracy:\t", "{:0.10f}".format(np.mean(bootstrapped_accuracies)),
          '\t', "{:0.10f}".format(np.mean(SMOTE_accuracies)))
    print("Average Precision:\t", "{:0.10f}".format(np.mean(bootstrapped_precisions)),
          '\t', "{:0.10f}".format(np.mean(SMOTE_precisions)))
    print("Average Recall:\t\t", "{:0.10f}".format(np.mean(bootstrapped_recalls)),
          '\t', "{:0.10f}".format(np.mean(SMOTE_recalls)))
    print("Average F1:\t\t", "{:0.10f}".format(np.mean(bootstrapped_f1s)),
          '\t', "{:0.10f}".format(np.mean(SMOTE_f1s)))


# So, the baseline model evaluation is done below. First, let's see how we can model original raw unprocessed data:

# In[ ]:


extrain_raw = pd.read_csv('Data/exoTrain.csv')


# In[ ]:


X_raw = extrain_raw.drop('LABEL', axis=1)
y_raw = extrain_raw.LABEL


# In[ ]:


model_evaluator(X_raw, y_raw, LinearSVC())


# In[ ]:


model_evaluator(X_raw, y_raw, LogisticRegression(n_jobs=-1))


# In[ ]:


model_evaluator(X_raw, y_raw, KNeighborsClassifier(n_jobs=-1))


# In[ ]:


model_evaluator(X_raw, y_raw, GaussianNB())


# In[ ]:


model_evaluator(X_raw, y_raw, RandomForestClassifier(n_jobs=-1))


# In[ ]:


model_evaluator(X_raw, y_raw, SVC())


# Now let's see how well we can fit a model using data that we have processed previously:

# In[ ]:


X = extrain.drop('LABEL', axis=1)
y = extrain.LABEL


# In[ ]:


model_evaluator(X, y, LinearSVC())


# In[ ]:


model_evaluator(X, y, LogisticRegression(n_jobs=-1))


# In[ ]:


model_evaluator(X, y, KNeighborsClassifier(n_jobs=-1))


# In[ ]:


model_evaluator(X, y, GaussianNB())


# In[ ]:


model_evaluator(X, y, RandomForestClassifier(n_jobs=-1))


# ![img](http://www.ragan.com/Resource.ashx?sn=Attention_Grabbing_Ways_to_Open_Blog_Posts)

# In[ ]:


# model_evaluator(X, y, SVC()) # This one takes too long to train and the previous results are abysmal, so I will skip it.


# It looks like K-nearest neighbors classifier works well for both processed and minimally processed data. The fit improved from f1 score of 0.149 from the KNN trained on raw bootstrapped data to f1 score of 0.576 from the KNN trained on minimally processed data expanded with SMOTE. Let's see if we could improve our model further.  
# 
# ## Improving the model  
# 
# If exoplanets exist around any given star, they should revolve around the star with frequencies far smaller than flux noise and other electromagnetic phenomena affecting flux. Therefore, let's consider analyzing flux frequency spectrum rather than raw flux data. In theory, the presence of exoplanets should contribute to lower frequencies in the spectrum making them good features to use for identification of stars with potential exoplanets.

# In[ ]:


import scipy


# In[ ]:


def spectrum_getter(X):
    Spectrum = scipy.fft(X, n=X.size)
    return np.abs(Spectrum)


# In[ ]:


X_train = extrain.drop('LABEL', axis=1)
y_train = extrain.LABEL

X_test = extest.drop('LABEL', axis=1)
y_test = extest.LABEL


# In[ ]:


new_X_train = X_train.apply(spectrum_getter,axis=1)
new_X_test = X_test.apply(spectrum_getter,axis=1)


# In[ ]:


new_X_train.head()


# In[ ]:


# Segregate data for desigining the model and for the final test
y = y_train
X = new_X_train

y_final_test = y_test
X_final_test = new_X_test


# In[ ]:


df = X.join(y)
i = 13
spec1 = df[df.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
freq = np.arange(len(spec1)) * (1/(36.0*60.0)) # Sampling frequency is 1 frame per ~36 minutes, or about 0.00046 Hz
plt.figure(figsize=(15,5))
plt.title('Frequency spectrum of processed flux of star {} with confirmed exoplanets (removed upper outliers)'
          .format(i+1))
plt.ylabel('Unitless flux')
plt.xlabel('Frequency, Hz')
plt.plot(freq, spec1)


# Since [frequency spectra are symmetric](https://dsp.stackexchange.com/questions/4825/why-is-the-fft-mirrored), we need to remove half of all spectra:

# In[ ]:


X = X.iloc[:,:(X.shape[1]//2)]
X_final_test = X_final_test.iloc[:,:(X_final_test.shape[1]//2)]


# Let's see what spectra look like for stars:

# In[ ]:


# Obtaining flux frequency spectra for several stars without exoplanets from the train data:
df = X.join(y)
for i in [0, 999, 1999, 2999, 3999, 4999]:
    spec1 = df[df.LABEL == 1].drop('LABEL', axis=1).iloc[i,:]
    freq = np.arange(len(spec1)) * (1/(36.0*60.0)) # Sampling frequency is 1 frame per ~36 minutes, or about 0.00046 Hz
    plt.figure(figsize=(15,5))
    plt.title('Frequency spectrum of processed flux of star {} with NO confirmed exoplanets (removed upper outliers)'
              .format(i+1))
    plt.ylabel('Unitless flux')
    plt.xlabel('Frequency, Hz')
    plt.plot(freq, spec1)


# In[ ]:


# Obtaining flux frequency spectra for several stars with exoplanets from the train data:
df = X.join(y)
for i in [0, 9, 14, 19, 24, 29]:
    spec1 = df[df.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
    freq = np.arange(len(spec1)) * (1/(36.0*60.0)) # Sampling frequency is 1 frame per ~36 minutes, or about 0.00046 Hz
    plt.figure(figsize=(15,5))
    plt.title('Frequency spectrum of processed flux of star {} WITH confirmed exoplanets (removed upper outliers)'
              .format(i+1))
    plt.ylabel('Unitless flux')
    plt.xlabel('Frequency, Hz')
    plt.plot(freq, spec1)


# There is a noticeable bump on the left side of spectra for stars with exoplanets, so we may have a chance at improving our models, after all!  
# But first, let's do a little housekeeping.

# In[ ]:


X.columns


# Let's convert column names to frequencies they represent.

# In[ ]:


X_columns = np.arange(len(X.columns))
X_columns = X_columns * (1.0/(36.0*60.0)) # sampling frequency of our data
X.columns = X_columns
X_final_test.columns = X_columns


# In[ ]:


X.columns


# Now, let's do some modeling with our new and improved features!

# ## Modeling

# In[ ]:


model_evaluator(X, y, LinearSVC())


# In[ ]:


model_evaluator(X, y, LogisticRegression(n_jobs=-1))


# In[ ]:


model_evaluator(X, y, KNeighborsClassifier(n_jobs=-1))


# In[ ]:


model_evaluator(X, y, GaussianNB())


# In[ ]:


model_evaluator(X, y, RandomForestClassifier(n_jobs=-1))


# In[ ]:


model_evaluator(X, y, SVC())


# No huge improvements, although having a precision of 0.7 on KNN model trained with bootstrapped data is very nice considering limited resources. Let's see if normalizing spectra can make any positive difference:

# In[ ]:


from sklearn.preprocessing import normalize


# In[ ]:


X = pd.DataFrame(normalize(X))
X_final_test = pd.DataFrame(normalize(X_final_test))


# In[ ]:


# Obtaining flux frequency spectra for several stars with exoplanets from the train data:
df = X.join(y)
for i in [0, 9, 14, 19, 24, 29]:
    spec1 = df[df.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]
    freq = np.arange(len(spec1)) * (1/(36.0*60.0)) # Sampling frequency is 1 frame per ~36 minutes, or about 0.00046 Hz
    plt.figure(figsize=(15,5))
    plt.title('Frequency spectrum of processed flux of star {} WITH confirmed exoplanets (removed upper outliers)'
              .format(i+1))
    plt.ylabel('Unitless flux')
    plt.xlabel('Frequency, Hz')
    plt.plot(freq, spec1)


# In[ ]:


model_evaluator(X, y, LinearSVC())


# In[ ]:


model_evaluator(X, y, LogisticRegression(n_jobs=-1))


# In[ ]:


model_evaluator(X, y, KNeighborsClassifier(n_jobs=-1))


# In[ ]:


model_evaluator(X, y, GaussianNB())


# In[ ]:


model_evaluator(X, y, RandomForestClassifier(n_jobs=-1))


# In[ ]:


model_evaluator(X, y, SVC())


# Looks like we found our model to optimize! We can now look closely at Linear Support Vector Classification and find most optimal parameters for it via randomized and grid searches. Perhaps, we can improve the recall rate.  
# ## Linear SVC Optimization through Randomized and Grid Searches

# Let's see if the Randomized Search works with our data without taking it through SMOTE process. For that I will be setting `class_weight` parameter in `LinearSVC` classifier to `'balanced'`. Also, I am unsure how to tell Randomized Search that I am looking for LABEL = 2, so to avoid any confusion, I will convert all 1s to 0s and all 2s to 1s:

# In[ ]:


y_new = y - 1


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# develop your "tuned parameters"

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print(__doc__)

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y_new, test_size=0.5, stratify=y)

# Set the parameters by cross-validation
tuned_parameters = [{'penalty': ['l2'], #'l1'],
              'loss': ['hinge'],
              'dual': [True],
              'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
              'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
              'fit_intercept': [True, False],
              'intercept_scaling': [0.01, 0.1, 1, 10, 100],
              'class_weight': ['balanced'],
              'verbose': [0],
              'random_state': [None],
              'max_iter': [10, 100, 1000, 10000, 100000]},
                    {'penalty': ['l2'], #'l1'],
              'loss': ['squared_hinge'],
              'dual': [True, False],
              'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
              'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
              'fit_intercept': [True, False],
              'intercept_scaling': [0.01, 0.1, 1, 10, 100],
              'class_weight': ['balanced'],
              'verbose': [0],
              'random_state': [None],
              'max_iter': [10, 100, 1000, 10000, 100000]},
                   {'penalty': ['l1'],
              'loss': ['squared_hinge'], #'hinge'],
              'dual': [False],
              'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
              'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
              'fit_intercept': [True, False],
              'intercept_scaling': [0.01, 0.1, 1, 10, 100],
              'class_weight': ['balanced'],
              'verbose': [0],
              'random_state': [None],
              'max_iter': [10, 100, 1000, 10000, 100000]}]

model_scores = ['precision', 'recall', 'f1']

for model_score in model_scores:
    print("# Tuning hyper-parameters for %s" % model_score)
    print()
    for tuned_parameter in tuned_parameters:
        clf = RandomizedSearchCV(LinearSVC(), tuned_parameter, cv=3, scoring=model_score, n_jobs=-1)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


# Let's see if we can do better with SMOTE data balancing. This time `class_weight` is going to be `None` since the data will be balanced through synthetic data generation:

# In[ ]:


def SMOTE_synthesizer(X, y):
        sm = SMOTE(ratio = 1.0)
        X, y = sm.fit_sample(X, y)
        return X, y


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# develop your "tuned parameters"

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print(__doc__)

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y_new, test_size=0.5, stratify=y)

X_train, y_train = SMOTE_synthesizer(X_train, y_train)

# Set the parameters by cross-validation
tuned_parameters = [{'penalty': ['l2'], #'l1'],
              'loss': ['hinge'],
              'dual': [True],
              'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
              'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
              'fit_intercept': [True, False],
              'intercept_scaling': [0.01, 0.1, 1, 10, 100],
              'class_weight': [None],
              'verbose': [0],
              'random_state': [None],
              'max_iter': [10, 100, 1000, 10000, 100000]},
                    {'penalty': ['l2'], #'l1'],
              'loss': ['squared_hinge'],
              'dual': [True, False],
              'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
              'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
              'fit_intercept': [True, False],
              'intercept_scaling': [0.01, 0.1, 1, 10, 100],
              'class_weight': [None],
              'verbose': [0],
              'random_state': [None],
              'max_iter': [10, 100, 1000, 10000, 100000]},
                   {'penalty': ['l1'],
              'loss': ['squared_hinge'], #'hinge'],
              'dual': [False],
              'tol': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
              'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
              'fit_intercept': [True, False],
              'intercept_scaling': [0.01, 0.1, 1, 10, 100],
              'class_weight': [None],
              'verbose': [0],
              'random_state': [None],
              'max_iter': [10, 100, 1000, 10000, 100000]}]

model_scores = ['precision', 'recall', 'f1']

for model_score in model_scores:
    print("# Tuning hyper-parameters for %s" % model_score)
    print()
    for tuned_parameter in tuned_parameters:
        clf = RandomizedSearchCV(LinearSVC(), tuned_parameter, cv=3, scoring=model_score, n_jobs=-1)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


# So at the end of the day, it still looks like training on synthetic data works better than balancing. For models trained with SMOTE data, my best `precision`, `recall`, and `f1` scores were `1.00`, `0.47`, and `0.64`, respectively. For models that try to balance training data, my best `precision`, `recall`, and `f1` scores were `1.00`, `0.37`, and `0.54`, respectively. The best scores for SMOTE trained models were achieved with the following parameters:
# ```python
# LinearSVC(
#     C=100.0,
#     class_weight=None,
#     dual=True,
#     fit_intercept=True,
#     intercept_scaling=10,
#     loss='hinge',
#     max_iter=100000,
#     multi_class='ovr',
#     penalty='l2',
#     random_state=None,
#     tol=0.0001,
#     verbose=0)```
# Therefore, we shall now proceed with the Grid Search to zoom in on possible settings around these parameters:

# In[ ]:


from sklearn.grid_search import GridSearchCV

# develop your "tuned parameters"

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

print(__doc__)

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y_new, test_size=0.5, stratify=y)

X_train, y_train = SMOTE_synthesizer(X_train, y_train)

# Set the parameters by cross-validation
tuned_parameters = {'penalty': ['l2'],
              'loss': ['hinge'],
              'dual': [True],
              'tol': np.arange(0.00008, .00014, 0.00002),
              'C': list(range(100, 220, 20)),
              'fit_intercept': [True],
              'intercept_scaling': np.arange(10, 130, 30),
              'class_weight': [None],
              'verbose': [0],
              'random_state': [None]}
#               'max_iter': np.arange(10000, 130000, 30000)}

# LinearSVC(
#     C=100.0,
#     class_weight=None,
#     dual=True,
#     fit_intercept=True,
#     intercept_scaling=10,
#     loss='hinge',
#     max_iter=100000,
#     multi_class='ovr',
#     penalty='l2',
#     random_state=None,
#     tol=0.0001,
#     verbose=0)

model_scores = ['precision', 'recall', 'f1']

for model_score in model_scores:
    print("# Tuning hyper-parameters for %s" % model_score)
    print()

    clf = GridSearchCV(LinearSVC(), tuned_parameters, cv=3, scoring=model_score, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


# Unfortunately, none of the parameter searches yielded results better than our initial default LinearSVC() model. Therefore, we shall proceed and create our final model uising default parameters of LinearSVC():

# ## Final Model

# In[ ]:


X_sm, y_sm = SMOTE_synthesizer(X, y)


# In[ ]:


final_model = LinearSVC()
final_model.fit(X_sm, y_sm)


# In[ ]:


y_pred = final_model.predict(X_final_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_final_test, y_pred))


# This is a good precision and recall, especially in comparison with earlier baseline models. It will be interesting to see the adoption of such approach to star flux analysis in further K2 campaigns.

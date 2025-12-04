# This programs is aiming to distinguish between Plasmodium infected and uninfected mosquitoes;

#%%

# Importing all modules

import os
import io
import ast
import itertools
import collections
import json
from time import time
from tqdm import tqdm 

import numpy as np # for mathematical computation
import pandas as pd # for mathematical computation

import scipy.stats as stats
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

from random import randint
from collections import Counter 

import pickle

from sklearn.model_selection import (
    KFold,
    train_test_split, 
    StratifiedShuffleSplit, 
    cross_val_score, 
    RandomizedSearchCV
)

import sklearn.metrics as metrics
from sklearn.metrics import(
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support, 
    roc_curve
)

from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(
    context = "paper",
    style = "white",
    palette = "deep",
    font_scale = 2.0,
    color_codes = True,
    rc = ({"font.family": "Dejavu Sans"})
    )

%matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]

#%%

# define a convenient plotting function (confusion matrix)

def plot_confusion_matrix(
    cm, 
    classes,
    normalise = True,
    text = False,
    title = 'Confusion matrix',
    xrotation = 0,
    yrotation = 0,
    cmap = plt.cm.Blues,
    printout = False
    ):
    """
    This function prints and plots the confusion matrix.
    Normalisation can be applied by setting 'normalise=True'.
    """

    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        if printout:
            print("Normalized confusion matrix")
        
    else:
        if printout:
            print('Confusion matrix')

    if printout:
        print(cm)

    plt.figure(figsize=(6, 4))
    plt.imshow(
        cm, 
        interpolation = 'nearest', 
        cmap = cmap, 
        vmin = 0.2, 
        vmax = 1.0
        )

    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.set_ylim(len(classes)-0.5, -0.5)
    plt.xticks(tick_marks, classes, rotation=xrotation)
    plt.yticks(tick_marks, classes, rotation=yrotation)

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1])):
            plt.text(
                j, 
                i, 
                format(cm[i, j], fmt), 
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
                )

    plt.tight_layout()
    plt.ylabel('True label', weight = 'bold')
    plt.xlabel('Predicted label', weight = 'bold')


#%%

# load data
df = pd.read_csv(
    r"C:\Mannu\Projects\Roger_ac_ag\data\male_data_lab.dat", 
    delimiter = '\t'
    )
df

#%%

# data shape
print(df.shape)

# Checking class distribution abd correlation in the data
print('Age', Counter(df["Cat3"]))
print('Species', Counter(df["Cat1"]))

#%% 

# Change age from string into a number (Chronological age)

Age = []

for row in df['Cat3']:

    if row == '01D':
        Age.append(1)
    
    elif row == '02D':
        Age.append(2)
    
    elif row == '03D':
        Age.append(3)

    elif row == '04D':
        Age.append(4)

    elif row == '05D':
        Age.append(5)

    elif row == '06D':
        Age.append(6)

    elif row == '07D':
        Age.append(7)

    elif row == '08D':
        Age.append(8)

    elif row == '09D':
        Age.append(9)

    elif row == '10D':
        Age.append(10)

    elif row == '11D':
        Age.append(11)

    elif row == '12D':
        Age.append(12)

    elif row == '13D':
        Age.append(13)

    elif row == '14D':
        Age.append(14)

    elif row == '15D':
        Age.append(15)
    elif row == '16D':
        Age.append(16)
    else:
        Age.append(17)

df['Cat3'] = Age


#%%

# Change the age into age groups

age_group = []

for row in df['Cat3']:

    if row <= 4:
        age_group.append('1-4')
    elif row > 4 and row <= 10:
        age_group.append('5-10')
    else:
        age_group.append('11-17')

df['Cat3'] = age_group

#%%

# Drop unused columns
train_df = df.drop(['Cat2', 'Cat4', 'Cat5', 'Cat6', 'StoTime'], axis = 1)

# Rename columns names forlabels to be more informative
train_df.rename(columns = {'Cat1':'Species', 'Cat3':'Age'}, inplace = True) 
train_df

#%% 
# Count the number of species and age groups

print('number of species : {}'.format(Counter(train_df['Species'])))
print('number of species : {}'.format(Counter(train_df['Age'])))

#%%

# Training and Test Data Splitting
# Stratify by species and age

train_set, test_set = train_test_split(
                            train_df, 
                            stratify = train_df[["Species", "Age"]], 
                            test_size = 0.2, 
                            shuffle = True, 
                            random_state = 42
                            )

# Count the number of species and age groups

print('number of species : {}'.format(Counter(train_set['Species'])))
print('number of species : {}'.format(Counter(train_set['Age'])))

# Creating a feature matrix and labels vector

# train set
X_train = np.asarray(train_set.iloc[:,2:]) # feature matrix
y_species = np.asarray(train_set['Species'])
y_age = np.asarray(train_set['Age'])

# test set

X_test = np.asarray(test_set.iloc[:,2:]) # feature matrix
y_test_species = np.asarray(test_set['Species'])
y_test_age = np.asarray(test_set['Age'])

print('The shape of X train set : {}'.format(X_train.shape))
print('The shape of y_species train set : {}'.format(y_species.shape))
print('The shape of y_age train set : {}'.format(y_age.shape))
print('The shape of X test set : {}'.format(X_test.shape))
print('The shape of y_species test set : {}'.format(y_test_species.shape))
print('The shape of y_age test set : {}'.format(y_test_age.shape))

# Classes are not balanced
# undersample classes for species

# define undersampling strategy
rus = RandomUnderSampler(random_state = 42)

# for species
X_species_train, y_species_train = rus.fit_resample(X_train, y_species)
print('Number of resampled species : {}'.format(Counter(y_species_train)))

# for age 
X_age_train, y_age_train = rus.fit_resample(X_train, y_age)
print('Number of resampled age : {}'.format(Counter(y_age_train)))

#%%

# standardisation 
# for species
sp_scl = StandardScaler().fit(X = X_species_train)
X_species_train  = sp_scl.transform(X = X_species_train)

# for age
ag_scl = StandardScaler().fit(X = X_age_train)
X_age_train  = ag_scl.transform(X = X_age_train)


#%%

# big LOOP for species prediction
# TUNNING THE SELECTED MODEL

# Set validation procedure
num_folds = 5 # split training set into 5 parts for validation
num_rounds = 5 # increase this to 5 or 10 once code is bug-free
seed = 42 # pick any integer. This ensures reproducibility of the tests
random_seed = np.random.randint(0, 81478)

features = train_set.iloc[:,2:]

scoring = 'accuracy' # score model accuracy

# cross validation strategy
kf = KFold(
        n_splits = num_folds, 
        shuffle = True, 
        random_state = random_seed
        )

# prepare matrices of results
kf_results = pd.DataFrame() # model parameters and global accuracy score
kf_per_class_results = [] # per class accuracy scores

species_predicted, species_true = [], [] # save species predicted and tre species values for each loop
start = time()

# Specify model
# Define the XGBoost model for species prediction
species_model = XGBClassifier()

# set hyparameter

estimators = [500, 1000]
rate = [0.05, 0.10, 0.15, 0.20, 0.30]
depth = [2, 3, 4, 5, 6, 8, 10, 12, 15]
child_weight = [1, 3, 5, 7]
gamma = [0.0, 0.1, 0.2, 0.3, 0.4]
bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

random_grid = {
              'n_estimators': estimators, 
              'learning_rate': rate, 
              'max_depth': depth,
              'min_child_weight': child_weight, 
              'gamma': gamma, 
              'colsample_bytree': bytree
              }

# under-sample over-represented classes (Negative class)

# for round in range (num_rounds):
#     SEED = np.random.randint(0, 81478)


# cross validation and splitting of the validation set
for train_index, test_index in kf.split(X_species_train, y_species_train):
    X_train_set, X_test_set = X_species_train[train_index], X_species_train[test_index]
    y_train_species, y_val_species = y_species_train[train_index], y_species_train[test_index]
  

    print('The shape of X train set : {}'.format(X_train_set.shape))
    print('The shape of y train species : {}'.format(y_train_species.shape))
    print('The shape of X test set : {}'.format(X_test_set.shape))
    print('The shape of y test species : {}'.format(y_val_species.shape))



    # generate models using all combinations of settings

    # RANDOMSED GRID SEARCH
    # Random search of parameters, using 5 fold cross validation, 
    # search across 100 different combinations, and use all available cores

    n_iter_search = 10
    rsCV = RandomizedSearchCV(
                            verbose = 1,
                            estimator = species_model, 
                            param_distributions = random_grid, 
                            n_iter = n_iter_search, 
                            scoring = scoring, 
                            cv = kf, 
                            refit = True, 
                            n_jobs = -1
                            )
    
    rsCV_result = rsCV.fit(X_train_set, y_train_species)

    # print out results and give hyperparameter settings for best one
    means = rsCV_result.cv_results_['mean_test_score']
    stds = rsCV_result.cv_results_['std_test_score']
    params = rsCV_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%.2f (%.2f) with: %r" % (mean, stdev, param))

    # print best parameter settings
    print("Best: %.2f using %s" % (rsCV_result.best_score_,
                                rsCV_result.best_params_))

    # Insert the best parameters identified by randomized grid search into the base classifier
    species_classifier = species_model.set_params(**rsCV_result.best_params_)
    
    # Fit your models
    species_classifier.fit(X_train_set, y_train_species)

    # predict test instances 
    sp_predictions = species_classifier.predict(X_test_set)

    # zip all predictions for plotting averaged confusion matrix
    # species
    for predicted_sp, true_sp in zip(sp_predictions, y_val_species):
        species_predicted.append(predicted_sp)
        species_true.append(true_sp)

    # species local confusion matrix & classification report
    local_cm_species = confusion_matrix(y_val_species, sp_predictions)
    local_report_species = classification_report(y_val_species, sp_predictions)

    # append feauture importances
    local_feat_impces_species = pd.DataFrame(species_classifier.feature_importances_,
                                        index = features.columns).sort_values(by = 0, ascending = False)

    # summarizing results
    local_kf_results = pd.DataFrame(
                                [
                                    ("Accuracy_species", accuracy_score(y_val_species, sp_predictions)), 
                                    ("TRAIN",str(train_index)), 
                                    ("TEST",str(test_index)), 
                                    ("CM", local_cm_species), 
                                    ("Classification report", local_report_species), 
                                    ("y_test", y_val_species),
                                    ("Feature importances", local_feat_impces_species.to_dict())
                                ]
                            ).T
    
    local_kf_results.columns = local_kf_results.iloc[0]
    local_kf_results = local_kf_results[1:]
    kf_results = pd.concat(
                        [kf_results, local_kf_results],
                        axis = 0,
                        join = 'outer'
                        ).reset_index(drop = True)

    # per class accuracy
    local_support = precision_recall_fscore_support(y_val_species, sp_predictions)[3]
    local_acc = np.diag(local_cm_species)/local_support
    kf_per_class_results.append(local_acc)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
elapsed / 60, elapsed))


#%%

# save species classifier to disk

with open(
    'C:\Mannu\Projects\Roger_ac_ag\species_classifier.pkl', 'wb'
    ) as fid:
    pickle.dump(species_classifier, fid)

#%%

# plot averaged confusion matrix for training
averaged_CM = confusion_matrix(species_true, species_predicted)
sp_classes = np.unique(np.sort(y_val_species))

sns.set(
    context = "paper",
    style = "white",
    palette = "deep",
    font_scale = 2.0,
    color_codes = True,
    rc = ({"font.family": "Dejavu Sans"})
    )

plot_confusion_matrix(averaged_CM, sp_classes)
plt.savefig(
    ("C:\Mannu\Projects\Roger_ac_ag\_averaged_CM_species.png"), 
    dpi = 500, 
    bbox_inches = "tight"
    )

#%%

# Results
kf_results.to_csv("C:\Mannu\Projects\Roger_ac_ag\crf_kfCV_record_species.csv", index=False)
kf_results = pd.read_csv("C:\Mannu\Projects\Roger_ac_ag\crf_kfCV_record_species.csv")

# Accuracy distribution
crf_acc_distrib = kf_results["Accuracy_species"]
crf_acc_distrib.columns = ["Accuracy_species"]

crf_acc_distrib.to_csv(
            "C:\Mannu\Projects\Roger_ac_ag\crf_acc_distrib_species.csv", 
            header =True, 
            index = False
            )
    
crf_acc_distrib = pd.read_csv("C:\Mannu\Projects\Roger_ac_ag\crf_acc_distrib_species.csv")
crf_acc_distrib = np.round(crf_acc_distrib, 2)
print(crf_acc_distrib)

#%%

# plotting accuracy distribution
plt.figure(figsize=(2.25,3))
sns.distplot(crf_acc_distrib, kde=False, bins=12)
# plt.savefig("lgr_acc_distrib.png", bbox_inches="tight")

#%%

# class distribution 

crf_per_class_acc_distrib = pd.DataFrame(kf_per_class_results, columns = sp_classes)
crf_per_class_acc_distrib.dropna().to_csv("C:\Mannu\Projects\Roger_ac_ag\crf_per_class_acc_distrib_species.csv")
crf_per_class_acc_distrib = pd.read_csv("C:\Mannu\Projects\Roger_ac_ag\crf_per_class_acc_distrib_species.csv", index_col=0)
crf_per_class_acc_distrib = np.round(crf_per_class_acc_distrib, 2)
crf_per_class_acc_distrib_describe = crf_per_class_acc_distrib.describe()
crf_per_class_acc_distrib_describe.to_csv("C:\Mannu\Projects\Roger_ac_ag\crf_per_class_acc_distrib_species.csv")


#%%

# plotting class distribution
lgr_per_class_acc_distrib = pd.melt(
                                crf_per_class_acc_distrib, 
                                var_name = "species new"
                                )

plt.figure(figsize = (6,4))
sns.set(
    context = "paper",
    style = "white",
    palette = "deep",
    font_scale = 2.0,
    color_codes = True,
    rc = ({"font.family": "Dejavu Sans"})
    )

sns.violinplot(
    x = "species new", 
    y = "value", 
    cut = 2, 
    data = lgr_per_class_acc_distrib
    )

sns.despine(left = True)
plt.xticks(rotation = 0, ha = "right")
plt.xticks()
plt.yticks(np.arange(0.2, 1.0 + .05, step = 0.2))
plt.xlabel(" ")
plt.ylabel("Prediction accuracy", weight = "bold")

plt.savefig(
    ("C:\Mannu\Projects\Roger_ac_ag\per_class_acc_distrib_species.png"), 
    dpi = 500, 
    bbox_inches = "tight"
    )

#%% 
 
# Feature Importances
# make this into bar with error bars across all best models

rskf_results = pd.read_csv(
                    "C:\Mannu\Projects\Roger_ac_ag\crf_kfCV_record_species.csv"
                    )

# Create a DataFrame from the first set of feature importances
all_featimp = pd.DataFrame(ast.literal_eval(rskf_results["Feature importances"][0]))

# Iterate through the rest of the feature importance sets
for featimp in rskf_results["Feature importances"][1:]:
    # Create a DataFrame from the current set of feature importances
    featimp = pd.DataFrame(ast.literal_eval(featimp))
    # Concatinate the current feature importances as new column to the existing DataFrame
    all_featimp = pd.concat(
                        [all_featimp, featimp], 
                        axis = 1, 
                        ignore_index = True
                        )

all_featimp["mean"] = all_featimp.mean(axis = 1)
all_featimp["sem"] = all_featimp.sem(axis = 1)
all_featimp.sort_values(by = "mean", inplace = True)

featimp_global_mean = all_featimp["mean"].mean()
featimp_global_sem = all_featimp["mean"].sem()

sns.set(context="paper",
    style="white",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})


fig = all_featimp["mean"][-50:].plot(
                                    figsize = (3, 14),
                                    kind = "barh",
                                    # orientation = 'vertical',
                                    legend = False,
                                    xerr = all_featimp["sem"],
                                    ecolor = 'k'
                                    )
plt.xlabel("Feature importance", weight = 'bold')
# plt.axvspan(xmin=0, xmax=featimp_global_mean+3*featimp_global_sem,facecolor='r', alpha=0.3)
# plt.axvline(x=featimp_global_mean, color="r", ls="--", dash_capstyle="butt")
sns.despine()

# Add mean accuracy of best models to plots
plt.annotate("Average Accuracy:\n{0:.3f} ± {1:.3f}".format(crf_acc_distrib.mean()[
             0], crf_acc_distrib.sem()[0]), xy=(0.06, 0), color="k")

plt.savefig(
    ("C:\Mannu\Projects\Roger_ac_ag\_feature_impces_species.png"), 
    dpi = 500, 
    bbox_inches = "tight"
    )

#%%

# Predict validation data
# Transform data using the mean and standard deviation from the model training data

X_species_val = sp_scl.transform(X = X_test)

# Predict test set

y_species_pred = species_classifier.predict(X_species_val)

species_accuracy = accuracy_score(y_test_species, y_species_pred)
print("Accuracy: %.2f%%" % (species_accuracy * 100.0))

#%%

# Plotting confusion matrix for X_val 

cm_species = confusion_matrix(y_test_species, y_species_pred)
sns.set(
    context = "paper",
    style = "white",
    palette = "deep",
    font_scale = 2.0,
    color_codes = True,
    rc = ({"font.family": "Dejavu Sans"})
    )

plot_confusion_matrix(cm_species, classes = sp_classes)
plt.savefig(("C:\Mannu\Projects\Roger_ac_ag\CM_species.png"), dpi = 500, bbox_inches="tight")

#%%

# Summarising precision, f_score, and recall for the validation set

cr_species = classification_report(y_test_species, y_species_pred)
print('Classification report : {}'.format(cr_species))

# save classification report to disk as a csv

cr_species = pd.read_fwf(io.StringIO(cr_species), header=0)
cr_species = cr_species.iloc[0:]
cr_species.to_csv("C:\Mannu\Projects\Roger_ac_ag\classification_report_species.csv")

#%%

# big LOOP for age prediction
# TUNNING THE SELECTED MODEL

# prepare matrices of results
kf_results_age = pd.DataFrame() # model parameters and global accuracy score
kf_per_class_results_age = [] # per class accuracy scores

age_predicted, age_true = [], [] # save age predicted and tre age values for each loop

start = time()

# Specify model

# Define the XGBoost model for age prediction
age_model = XGBClassifier(objective = 'multi:softmax')
age_class_labels = ['1-4', '5-10', '11-17']

# for round in range (num_rounds):
#     SEED = np.random.randint(0, 81478)


# cross validation and splitting of the validation set
for train_index_age, test_index_age in kf.split(X_age_train, y_age_train):
    X_train_set_age, X_test_set_age = X_age_train[train_index_age], X_age_train[test_index_age]
    y_train_age, y_val_age = y_age_train[train_index_age], y_age_train[test_index_age]


    print('The shape of X train set : {}'.format(X_train_set_age.shape))
    print('The shape of y train age : {}'.format(y_train_age.shape))
    print('The shape of X test set : {}'.format(X_test_set_age.shape))
    print('The shape of y test age : {}'.format(y_val_age.shape))


    # generate models using all combinations of settings

    # RANDOMSED GRID SEARCH
    # Random search of parameters, using 5 fold cross validation, 
    # search across 100 different combinations, and use all available cores

    n_iter_search = 10
    rsCV_age = RandomizedSearchCV(
                            verbose = 1,
                            estimator = age_model, 
                            param_distributions = random_grid, 
                            n_iter = n_iter_search, 
                            scoring = scoring, 
                            cv = kf, 
                            refit = True, 
                            n_jobs = -1)
    
    rsCV_result_age = rsCV_age.fit(X_train_set_age, y_train_age)

    # print out results and give hyperparameter settings for best one
    means_age = rsCV_result_age.cv_results_['mean_test_score']
    stds_age = rsCV_result_age.cv_results_['std_test_score']
    params_age = rsCV_result_age.cv_results_['params']
    for mean_age, stdev_age, param_age in zip(means_age, stds_age, params_age):
        print("%.2f (%.2f) with: %r" % (mean_age, stdev_age, param_age))

    # print best parameter settings
    print("Best: %.2f using %s" % (rsCV_result_age.best_score_,
                                rsCV_result_age.best_params_))

    # Insert the best parameters identified by randomized grid search into the base classifier
    age_classifier = age_model.set_params(**rsCV_result_age.best_params_)
    
    # Fit your model
    age_classifier.fit(X_train_set_age, y_train_age)

    # predict test instance
    ag_predictions = age_classifier.predict(X_test_set_age)
    # print('age : {}'.format(ag_predictions))

    # zip all predictions for plotting averaged confusion matrix
    # age
    for predicted_ag, true_ag in zip(ag_predictions, y_val_age):
        age_predicted.append(predicted_ag)
        age_true.append(true_ag)

    # age local confusion matrix & classification report
    local_cm_age = confusion_matrix(y_val_age, ag_predictions, labels = age_class_labels)
    local_report_age = classification_report(y_val_age, ag_predictions, labels = age_class_labels)


    # append feauture importances
    local_feat_impces_age = pd.DataFrame(age_classifier.feature_importances_,
                                        index = features.columns).sort_values(by = 0, ascending = False)

    # summarizing results
    local_kf_results_age = pd.DataFrame(
                                    [
                                        ("Accuracy_age", accuracy_score(y_val_age, ag_predictions)),
                                        ("TRAIN",str(train_index_age)), 
                                        ("TEST",str(test_index_age)), 
                                        ("CM", local_cm_age), 
                                        ("Classification report", local_report_age), 
                                        ("y_test", y_val_age),
                                        ("Feature importances", local_feat_impces_age.to_dict())
                                        ]
                                    ).T
    
    local_kf_results_age.columns = local_kf_results_age.iloc[0]
    local_kf_results_age = local_kf_results_age[1:]
    kf_results_age = pd.concat(
                            [kf_results_age, local_kf_results_age],
                            axis = 0,
                            join = 'outer'
                            ).reset_index(drop = True)

    # per class accuracy
    local_support_age = precision_recall_fscore_support(y_val_age, ag_predictions, labels = age_class_labels)[3]
    local_acc_age = np.diag(local_cm_age)/local_support_age
    kf_per_class_results_age.append(local_acc_age)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
elapsed / 60, elapsed))


#%%

# save age classifier to disk

with open(
    r'C:\Mannu\Projects\Roger_ac_ag\age_classifier.pkl', 'wb'
    ) as fid:
     pickle.dump(age_classifier, fid)

#%%


# plot averaged confusion matrix for training
averaged_CM_age = confusion_matrix(
                                age_true, 
                                age_predicted, 
                                labels = age_class_labels
                                )

age_classes = np.unique(np.sort(y_val_age))

plot_confusion_matrix(averaged_CM_age, age_class_labels)

plt.savefig(
    ("C:\Mannu\Projects\Roger_ac_ag\_averaged_CM_age.png"), 
    dpi = 500, 
    bbox_inches = "tight"
    )


#%%

# Age model Results
kf_results_age.to_csv("C:\Mannu\Projects\Roger_ac_ag\crf_kfCV_record_age.csv", index = False)
kf_results_age = pd.read_csv("C:\Mannu\Projects\Roger_ac_ag\crf_kfCV_record_age.csv")

# age accuracy distribution
crf_acc_distrib_age = kf_results_age["Accuracy_age"]
crf_acc_distrib_age.columns=["Accuracy_age"]

crf_acc_distrib_age.to_csv(
                        "C:\Mannu\Projects\Roger_ac_ag\crf_acc_distrib_age.csv", 
                        header = True, 
                        index = False
                        )

crf_acc_distrib_age = pd.read_csv("C:\Mannu\Projects\Roger_ac_ag\crf_acc_distrib_age.csv")
crf_acc_distrib_age = np.round(crf_acc_distrib_age, 2)
print(crf_acc_distrib_age)

#%%

# plotting age prediction accuracy distribution

plt.figure(figsize=(2.25,3))
sns.distplot(crf_acc_distrib_age, kde=False, bins=12)

#%%

# class age distribution 

crf_per_ageclass_acc_distrib = pd.DataFrame(kf_per_class_results_age, columns = age_class_labels)
crf_per_ageclass_acc_distrib.dropna().to_csv("C:\Mannu\Projects\Roger_ac_ag\crf_per_ageclass_acc_distrib.csv")
crf_per_ageclass_acc_distrib_age = pd.read_csv("C:\Mannu\Projects\Roger_ac_ag\crf_per_ageclass_acc_distrib.csv", index_col = 0)
crf_per_ageclass_acc_distrib = np.round(crf_per_ageclass_acc_distrib, 2)
crf_per_ageclass_acc_distrib_describe = crf_per_ageclass_acc_distrib.describe()
crf_per_ageclass_acc_distrib_describe.to_csv("C:\Mannu\Projects\Roger_ac_ag\crf_per_ageclass_acc_distrib.csv")


#%%

# plotting age class distribution

lgr_per_ageclass_acc_distrib = pd.melt(
                                    crf_per_ageclass_acc_distrib, 
                                    var_name = "age new"
                                    )


plt.figure(figsize=(6,4))

sns.violinplot(
            x = "age new", 
            y = "value", 
            cut = 2, 
            data = lgr_per_ageclass_acc_distrib
            )

sns.despine(left = True)
plt.xticks(rotation = 0, ha = "right")
plt.xticks()
plt.yticks(np.arange(0.2, 1.0 + .05, step = 0.2))
plt.xlabel(" ")
plt.ylabel("Prediction accuracy", weight = "bold")

plt.savefig(
        ("C:\Mannu\Projects\Roger_ac_ag\per_class_acc_distrib_age.png"), 
        dpi = 500, 
        bbox_inches = "tight"
        )


#%% 

# Feature Importances

# make this into bar with error bars across all best age models 

rskf_results_age = pd.read_csv("C:\Mannu\Projects\Roger_ac_ag\crf_kfCV_record_age.csv")

# Create a DataFrame from the first set of feature importances
all_featimp_age = pd.DataFrame(ast.literal_eval(rskf_results_age["Feature importances"][0]))

# Iterate through the rest of the feature importance sets
for featimp_age in rskf_results_age["Feature importances"][1:]:
    # Create a DataFrame from the current set of feature importances
    featimp_age = pd.DataFrame(ast.literal_eval(featimp_age))
    # Concatenate the current feature importances as new columns to the existing DataFrame
    all_featimp_age = pd.concat(
                            [all_featimp_age, featimp_age], 
                            axis = 1, 
                            ignore_index = True
                            )

all_featimp_age["mean"] = all_featimp_age.mean(axis=1)
all_featimp_age["sem"] = all_featimp_age.sem(axis=1)
all_featimp_age.sort_values(by="mean", inplace=True)

featimp_global_mean_age = all_featimp_age["mean"].mean()
featimp_global_sem_age = all_featimp_age["mean"].sem()

sns.set(context="paper",
    style="white",
    font_scale=2.0,
    rc={"font.family": "Dejavu Sans"})


fig_age = all_featimp_age["mean"][-50:].plot(
                                            figsize=(3, 14),
                                            kind="barh",
                                            # orientation = 'vertical',
                                            legend=False,
                                            xerr=all_featimp_age["sem"][-50:],
                                            ecolor='k'
                                            )
plt.xlabel("Feature importance", weight = 'bold')
# plt.axvspan(xmin=0, xmax=featimp_global_mean+3*featimp_global_sem,facecolor='r', alpha=0.3)
# plt.axvline(x=featimp_global_mean, color="r", ls="--", dash_capstyle="butt")
sns.despine()

# Add mean accuracy of best models to plots
plt.annotate("Average Accuracy:\n{0:.3f} ± {1:.3f}".format(crf_acc_distrib_age.mean()[
             0], crf_acc_distrib_age.sem()[0]), xy=(0.06, 0), color="k")

plt.savefig(
        ("C:\Mannu\Projects\Roger_ac_ag\_feature_impces_age.png"), 
        dpi = 500, 
        bbox_inches = "tight"
        )

#%%

# Predict age

# Transform data using the mean and standard deviation from the model training data

X_age_val = ag_scl.transform(X = X_test)

# Predict test set

y_age_pred = age_classifier.predict(X_age_val)

age_accuracy = accuracy_score(y_test_age, y_age_pred)
print("Accuracy: %.2f%%" % (age_accuracy * 100.0))

#%%

# Plotting confusion matrix for age prediction 

cm_age = confusion_matrix(
                        y_test_age, 
                        y_age_pred, 
                        labels = age_class_labels
                        )

plot_confusion_matrix(cm_age, classes = age_class_labels)

plt.savefig(
    ("C:\Mannu\Projects\Roger_ac_ag\CM_age.png"), 
    dpi = 500, 
    bbox_inches = "tight"
    )

#%%

# Summarising precision, f_score, and recall for the age age prediction

cr_age = classification_report(
                            y_test_age, 
                            y_age_pred, 
                            labels = age_class_labels
                            )

print('Classification report : {}'.format(cr_age))

# save classification report to disk as a csv

cr_age = pd.read_fwf(io.StringIO(cr_age), header=0)
cr_age = cr_age.iloc[0:]
cr_age.to_csv("C:\Mannu\Projects\Roger_ac_ag\classification_report_age.csv")


#############################################################################
#############################################################################
#############################################################################


# %%
# Using Lab model to predict age and species from field data

# load data
field_df = pd.read_csv(
    r"C:\Mannu\Projects\Roger_ac_ag\data\male_data_field.dat", 
    delimiter = '\t'
    )
field_df

#%%

# data shape
print(field_df.shape)

# Checking class distribution abd correlation in the data
print('Age', Counter(field_df["Cat3"]))
print('Species', Counter(field_df["Cat1"]))

#%% 

# Change age from string into a number (Chronological age)

Age_field = []

for row in field_df['Cat3']:

    if row == '01D':
        Age_field.append(1)
    
    elif row == '02D':
        Age_field.append(2)
    
    elif row == '03D':
        Age_field.append(3)

    elif row == '04D':
        Age_field.append(4)

    elif row == '05D':
        Age_field.append(5)

    elif row == '06D':
        Age_field.append(6)

    elif row == '07D':
        Age_field.append(7)

    elif row == '08D':
        Age_field.append(8)

    elif row == '09D':
        Age_field.append(9)

    elif row == '10D':
        Age_field.append(10)

    elif row == '11D':
        Age_field.append(11)

    elif row == '12D':
        Age_field.append(12)

    elif row == '13D':
        Age_field.append(13)

    elif row == '14D':
        Age_field.append(14)

    elif row == '15D':
        Age_field.append(15)
    elif row == '16D':
        Age_field.append(16)
    else:
        Age_field.append(17)

field_df['Cat3'] = Age_field


#%%

# Change the age into age groups

age_group_field = []

for row in field_df['Cat3']:

    if row <= 4:
        age_group_field.append('1-4')
    elif row > 4 and row <= 10:
        age_group_field.append('5-10')
    else:
        age_group_field.append('11-17')

field_df['Cat3'] = age_group_field

#%%

# Drop unused columns
field_test_df = field_df.drop(['Cat2', 'Cat4', 'Cat5', 'Cat6', 'StoTime'], axis = 1)

# Rename columns names forlabels to be more informative
field_test_df.rename(columns = {'Cat1':'Species', 'Cat3':'Age'}, inplace = True) 
field_test_df

#%%
# Make predictions on semi-field data

field_x_val = np.asarray(field_test_df.iloc[:,2:]) # feature matrix
field_species_val = np.asarray(field_test_df['Species'])
field_age_val = np.asarray(field_test_df['Age'])

# scale data
field_x_val_scl  = sp_scl.transform(X = field_x_val)

#%%
# Prediction
# predict semi-field species

field_sp_val_pred = species_classifier.predict(field_x_val_scl)

field_sp_val_accuracy = accuracy_score(field_species_val, field_sp_val_pred)
print("Accuracy: %.2f%%" % (field_sp_val_accuracy * 100.0))

# Plotting confusion matrix for the prediction

cm_field_sp_val = confusion_matrix(field_species_val, field_sp_val_pred)
sns.set(
    context = "paper",
    style = "white",
    palette = "deep",
    font_scale = 2.0,
    color_codes = True,
    rc = ({"font.family": "Dejavu Sans"})
    )

plot_confusion_matrix(cm_field_sp_val, classes = sp_classes)
plt.savefig(
                (r"C:\Mannu\Projects\Roger_ac_ag\field_species_val_pred.png"), 
                dpi = 500, 
                bbox_inches="tight"
            )

#%%
# predict semi-field age

field_age_val_pred = age_classifier.predict(field_x_val_scl)

field_age_val_accuracy = accuracy_score(field_age_val, field_age_val_pred)
print("Accuracy: %.2f%%" % (field_age_val_accuracy * 100.0))

# Plotting confusion matrix for the prediction

cm_field_age_val = confusion_matrix(field_age_val, field_age_val_pred, labels = age_class_labels)
sns.set(
    context = "paper",
    style = "white",
    palette = "deep",
    font_scale = 2.0,
    color_codes = True,
    rc = ({"font.family": "Dejavu Sans"})
    )

plot_confusion_matrix(cm_field_age_val, classes = age_class_labels)
plt.savefig(
                (r"C:\Mannu\Projects\Roger_ac_ag\field_age_val_pred.png"), 
                dpi = 500, 
                bbox_inches="tight"
            )



# %%

# Count the number of species and age groups

print('number of species : {}'.format(Counter(field_test_df['Species'])))
print('number of species : {}'.format(Counter(field_test_df['Age'])))

field_train_set, field_test_set = train_test_split(
                            field_test_df, 
                            stratify = field_test_df[["Species", "Age"]], 
                            test_size = 0.85, 
                            shuffle = True, 
                            random_state = 42
                            )

# Count the number of species and age groups

print('number of species : {}'.format(Counter(field_train_set['Species'])))
print('number of species : {}'.format(Counter(field_train_set['Age'])))

# Creating a feature matrix and labels vector

# train set
field_fts_train = np.asarray(field_train_set.iloc[:,2:]) # feature matrix
field_species_train = np.asarray(field_train_set['Species'])
field_age_train = np.asarray(field_train_set['Age'])

# test set

field_fts_test = np.asarray(field_test_set.iloc[:,2:]) # feature matrix
field_test_species = np.asarray(field_test_set['Species'])
field_test_age = np.asarray(field_test_set['Age'])

print('The shape of X train set : {}'.format(field_fts_train.shape))
print('The shape of y train set : {}'.format(field_species_train.shape))
print('The shape of y train set : {}'.format(field_age_train.shape))
print('The shape of X test set : {}'.format(field_fts_test.shape))
print('The shape of y test set : {}'.format(field_test_species.shape))
print('The shape of y test set : {}'.format(field_test_age.shape))


# %%
# standardisation 

field_fts_train  = sp_scl.transform(X = field_fts_train)
field_fts_test  = sp_scl.transform(X = field_fts_test)

#%%

# Field species prediction

# transfer learning

classifier = XGBClassifier() 

transfer_model_species = classifier.fit(
                                field_fts_train, 
                                field_species_train, 
                                xgb_model = species_classifier.get_booster()
                                ) 

# Prediction
field_species_pred = transfer_model_species.predict(field_fts_test)

field_species_accuracy = accuracy_score(field_test_species, field_species_pred)
print("Accuracy: %.2f%%" % (field_species_accuracy * 100.0))

# Plotting confusion matrix for X_val 

cm_field_species = confusion_matrix(field_test_species, field_species_pred)
sns.set(
    context = "paper",
    style = "white",
    palette = "deep",
    font_scale = 2.0,
    color_codes = True,
    rc = ({"font.family": "Dejavu Sans"})
    )

plot_confusion_matrix(cm_field_species, classes = sp_classes)
plt.savefig(
                ("C:\Mannu\Projects\Roger_ac_ag\CM_field_TL_species.png"), 
                dpi = 500, 
                bbox_inches="tight"
            )


# Summarising precision, f_score, and recall for the validation set

cr_field_species = classification_report(field_test_species, field_species_pred)
print('Classification report : {}'.format(cr_field_species))

# save classification report to disk as a csv

cr_field_species = pd.read_fwf(io.StringIO(cr_field_species), header=0)
cr_field_species = cr_field_species.iloc[0:]
cr_field_species.to_csv("C:\Mannu\Projects\Roger_ac_ag\cr_field_TL_species.csv")

#%%

# Field age predictions

# transfer learning

classifier = XGBClassifier() 
transfer_model_age = classifier.fit(
                                field_fts_train, 
                                field_age_train, 
                                xgb_model = age_classifier.get_booster()
                                ) 

# prediction
field_age_pred = transfer_model_age.predict(field_fts_test)

field_age_accuracy = accuracy_score(field_test_age, field_age_pred)
print("Accuracy: %.2f%%" % (field_age_accuracy * 100.0))


# Plotting confusion matrix for age prediction 

cm_field_age = confusion_matrix(
                        field_test_age, 
                        field_age_pred, 
                        labels = age_class_labels
                        )

plot_confusion_matrix(cm_field_age, classes = age_class_labels)

plt.savefig(
    ("C:\Mannu\Projects\Roger_ac_ag\CM_field_TL_age.png"), 
    dpi = 500, 
    bbox_inches = "tight"
    )

# Summarising precision, f_score, and recall for the age age prediction

cr_field_age = classification_report(
                            field_test_age, 
                            field_age_pred, 
                            labels = age_class_labels
                            )

print('Classification report : {}'.format(cr_field_age))

# save classification report to disk as a csv

cr_field_age = pd.read_fwf(io.StringIO(cr_field_age), header=0)
cr_field_age = cr_field_age.iloc[0:]
cr_field_age.to_csv("C:\Mannu\Projects\Roger_ac_ag\cr_field_TL_age.csv")

# %%

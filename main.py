
import os
import statistics
import numpy

import pandas as pd
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.metrics

plt.style.use('ggplot')

### Import the python equivalent of here package in R
from pyhere import here
data = here('Input', 'churn_bank.csv')

# importing the dataset
df = pd.read_csv(data)
print(df)

##### identifying the churn rate among bank's customers #####

# resuming all the data contained in the data frame
print(df.info())

# var 'Exited' seems to be the target variable with which we can compute the churn rate. Hence, the variable is a dummy (either O or 1).
# We compute the churn rate by computing the average on the var 'Exited'
churn_rate = statistics.mean(df['Exited'])

#creating gender-specific dataframes
male_df = df[df['Gender'] == 'Male']
female_df = df[df['Gender'] == 'Female']

# comparing means computed on all df
male_basic_stats = male_df.describe()
female_basic_stats = female_df.describe()
overall_basic_stats = df.describe()



# storing basic stats dfs
dfs = [male_basic_stats, female_basic_stats, overall_basic_stats]

dfs_mean = []
for i in dfs:
    temp_df = i.loc['mean']
    dfs_mean.append(temp_df)

# merging mean overall and gender-specific df
df_mean_merged = dfs_mean[0].to_frame().merge(dfs_mean[1], how='left').merge(dfs_mean[2], how='left')

#########################################################################################################################
################################# Implementing a simple machine learning model (Knn model) ##########################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import pipeline


### preprocessing dataset
df.drop(['CustomerId', 'Surname', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember'], inplace=True, axis=1)

ord_enc = OrdinalEncoder()
df['Geography'] = ord_enc.fit_transform(df['Geography'])
df['Gender'] = ord_enc.fit_transform(df['Gender'])

Y = pd.DataFrame(df['Exited'])
X = df.drop('Exited', inplace=True, axis=1)
X = df

accur1 = []

for i in range(1,80):
    # Specifying the model to train and test
    pipe = pipeline.make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(i))
    # defining explanatory variables dataset, target variable dataset, train and test sets
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=53)
    # running the model on the train dataset
    pipe.fit(X_train, Y_train)
    Y_pred = pipe.predict(X_test)
    # model accuracy
    accuracy_temp = accuracy_score(Y_test, Y_pred)
    accur1.append(accuracy_temp)

### generating matplotlib graphs
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(np.arange(1,80), accur1);  # Plot some data on the axes.
ax.set_xlabel('Neighbors')
ax.set_ylabel('Accuracy')
fig.align_labels()

### Selecting the best model among the ones tested with different KNeighborsClassifier
data = [accur1.index(max(accur1)), max(accur1)]
best_model_select = pd.DataFrame([data], columns=["KNeighborsClassifier", "Accuracy"])

####################################################################################
### We implement the Knn ML model for a higher range of variables taken into account
####################################################################################

df = pd.read_csv(r'/Users/hugoc/Desktop/Academic/M2 QEA Dauphine/Python for data science/Project/Input/churn_bank.csv')

df.drop(['Surname', 'CustomerId'], inplace=True, axis=1)

ord_enc = OrdinalEncoder()
df['Geography'] = ord_enc.fit_transform(df['Geography'])
df['Gender'] = ord_enc.fit_transform(df['Gender'])

Y = pd.DataFrame(df['Exited'])
df.drop(['Exited'], inplace=True, axis=1)
X = df

accur2 = []

for i in range(1,80):
    # Specifying the model to train and test
    pipe = pipeline.make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(i))
    # defining explanatory variables dataset, target variable dataset, train and test sets
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=53)
    # running the model on the train dataset
    pipe.fit(X_train, Y_train)
    Y_pred = pipe.predict(X_test)
    # model accuracy
    accuracy_temp = accuracy_score(Y_test, Y_pred)
    accur2.append(accuracy_temp)

### generating matplotlib graphs
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(np.arange(1,80), accur2);  # Plot some data on the axes.
ax.set_xlabel('Neighbors')
ax.set_ylabel('Accuracy')
fig.align_labels()

### Plotting both accuracy graphs in order to compare respective models' performances
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(np.arange(1,80), accur1, accur2);  # Plot some data on the axes.
ax.set_xlabel('Neighbors')
ax.set_ylabel('Accuracy')
fig.align_labels()

### Selecting the best model among the ones tested with different KNeighborsClassifier
data = [accur2.index(max(accur2)), max(accur2)]
best_model_select = pd.DataFrame([data], columns=["KNeighborsClassifier", "Accuracy"])


### We run the best model in order to extract the ROC curve
pipe = pipeline.make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(accur2.index(max(accur2))))
# defining explanatory variables dataset, target variable dataset, train and test sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=53)
# running the model on the train dataset
pipe.fit(X_train, Y_train)
Y_pred = pipe.predict(X_test)
Y_pred_proba = pipe.predict_proba(X_test)
# model accuracy
print(accuracy_score(Y_test, Y_pred))

roc_auc = roc_auc_score(Y_test, Y_pred)
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba[:,1])
plt.figure()
plt.plot(fpr, tpr, label='ML Model (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ML_ROC')
plt.show()




















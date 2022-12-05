
import os
import statistics
import numpy

import pandas as pd
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()
print(cwd)

# importing the dataset
df = pd.read_csv(r'/Users/hugoc/Desktop/Academic/M2 QEA Dauphine/Python for data science/Project/Input/churn_bank.csv')
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
################################# Implementing a simple machine learning model ##########################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import pipeline

# preprocessing dataset
df.drop(['CustomerId', 'Surname', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember'], inplace=True, axis=1)

ord_enc = OrdinalEncoder()
df['Geography'] = ord_enc.fit_transform(df['Geography'])
df['Gender'] = ord_enc.fit_transform(df['Gender'])

Y = pd.DataFrame(df['Exited'])
X = df.drop('Exited', inplace=True, axis=1)
X = df

accur = []

for i in range(1,40):
    # Specifying the model to train and test
    pipe = pipeline.make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(i))
    # defining explanatory variables dataset, target variable dataset, train and test sets
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=53)
    # running the model on the train dataset
    pipe.fit(X_train, Y_train)
    Y_pred = pipe.predict(X_test)
    # model accuracy
    accuracy_temp = accuracy_score(Y_test, Y_pred)
    accur.append(accuracy_temp)

### generating matplotlib graphs














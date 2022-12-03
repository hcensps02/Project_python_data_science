
import os
import statistics
import numpy

import pandas as pd
from functools import reduce
import numpy as np


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



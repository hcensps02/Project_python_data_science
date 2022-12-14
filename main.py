
import os
import statistics

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
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



###################################################################################
###                                 Data analsis                                ###
###################################################################################

fig,ax=plt.subplots(figsize=[7,5])
sns.histplot(df,x="EstimatedSalary")
ax.set(title="Estimated Salary Distribution",xlabel="Estimated salary (annual)",ylabel="Frequency")


#interactions between variables

#####Gender
sns.displot(data=df, x="Age", hue="Geography", kind="kde",common_norm=False)
sns.displot(data=df, x="Age", hue="Gender", kind="kde",common_norm=False)
sns.displot(data=df, x="Age", hue="Gender", kind="kde",common_norm=False)

sns.catplot(data=df, x="Gender", y="EstimatedSalary", kind="box")

sns.countplot(x='NumOfProducts', hue='Gender', data=df, palette='hls') #trouver comment partager l'histogramme entre exited non exited

G_nb_products_gender = px.histogram(df, x='Gender', color='NumOfProducts', barnorm = 'fraction', barmode='relative',text_auto=True)
G_nb_products_gender.show()

G_CrCard = px.histogram(df, x='Gender', color='HasCrCard', barnorm = 'percent', barmode='relative',text_auto=True)
G_CrCard.show() #similar between genders

#####Geography

#Pie chart
counts = df['Geography'].value_counts()
df2 = pd.DataFrame({'Geography': counts},
                     index = ['France', 'Spain', 'Germany']
                   )
ax = df2.plot.pie(y='Geography', figsize=(5,5), autopct='%1.1f%%')
sns.displot(data=df, x="Balance", hue="Geography", kind="kde", common_norm=False) #grosse concentration en France et Spain autour de zero et negatif, Germany concentration en positif
sns.catplot(data=df, x="Geography", y="EstimatedSalary", kind="box")

##Geography
G_nb_products = px.histogram(df, x='Geography', color='NumOfProducts', barnorm = 'fraction', barmode='relative',text_auto=True).update_yaxes(categoryorder='total ascending')
G_nb_products.show()#similar shares of nb of products,large part of products 1 and 2

#Exit
df['Exited'].value_counts()
sns.countplot(x='Exited', data=df, palette='hls')
plt.show()
plt.savefig('count_plot')

#Exited vs non exited
#g=sns.FacetGrid(df, col="Exited")
#f=g.map_dataframe(sns.histplot, x="CreditScore") #exited pple have on average less credit score
sns.displot(data=df, x="CreditScore", hue="Exited", kind="kde",common_norm=False).set(xlabel='Credit Score')
sns.displot(data=df, x="EstimatedSalary", hue="Exited", kind="kde",common_norm=False).set(xlabel='Estimated Salary')
sns.displot(data=df, x="Balance", hue="Exited", kind="kde",common_norm=False).set(xlabel="Customer's bank account's balance")#common norm = false=> normalize each histogram independantly


sns.catplot(data=df, x="Exited", y="CreditScore", kind="box").set(ylabel='Credit Score')
sns.catplot(data=df, x="Exited", y="EstimatedSalary", kind="box").set(ylabel='Estimated Salary')
sns.catplot(data=df, x="Exited", y="Balance", kind="box").set(ylabel='Balance')

#socio geograhic factors
sns.countplot(x='Gender', hue='Exited', data=df, palette='hls')
sns.histplot(data=df, x="Geography", hue="Exited", multiple="stack")
sns.displot(data=df, x="Age", hue="Exited", kind="kde",common_norm=False)
sns.catplot(data=df, x="Exited", y="Age", kind="box")
#exited people have on average a higher balance, which may be due to the large number of zero-balance among the non exited persons
#s'explique par une plus grande distribution de 0 balance chez les non exited

df['Balance'].corr(df['EstimatedSalary'])  #correlation 12% between salary and balance, preditible
df['Balance'].corr(df['CreditScore'])


#####PROBIT MODEL IMPLEMENTATION
import statsmodels as sm
from statsmodels.discrete.discrete_model import Probit

#create dummy variables for categorical variables
cat_var=['Geography', 'Gender']
for var in cat_var:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df[var], prefix=var)
    data1=df.join(cat_list)
    df=data1
#cat_var=['Geography', 'Gender']
data_vars=df.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_var]

data_final=df[to_keep]
data_final.columns.values

Y = data_final["Exited"]
X = data_final.drop(['Exited','CustomerId', 'Surname'], axis=1)
#X = sm.add_constant(X)
model = Probit(Y, X)
probit_model = model.fit()
print(probit_model.summary())

f = open('myreg.tex', 'w')
f.write(probit_model.summary().as_latex())


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


###
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




















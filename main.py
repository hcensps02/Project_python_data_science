
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
male_basic_stats = male_df.describe(include = 'all').round(decimals=2)
female_basic_stats = female_df.describe(include = 'all').round(decimals=2)

overall_basic_stats = df.describe(include = 'all').drop(['Surname', 'CustomerId'], axis=1).round(decimals=2)
overall_basic_stats.plot()

#exporting basic statistics tables into latex tables
print(male_basic_stats.to_latex(index=False))
print(female_basic_stats.to_latex(index=False))
print(overall_basic_stats.drop(['Surname', 'CustomerId'], axis=1).to_latex(index=False))


# Generating corr matrix
modif_df = df.drop(['Surname', 'CustomerId'], axis=1)
corr_mat = modif_df.corr().round(decimals=2)
print(corr_mat.to_latex(index=False))




###################################################################################
##########                          Data analysis                         #########
###################################################################################


fig,ax=plt.subplots(figsize=[7,5])
sns.histplot(df,x="EstimatedSalary")
ax.set(title="Estimated Salary Distribution",xlabel="Estimated salary (annual)",ylabel="Frequency")


#############Variable interactions

#####Gender

#Age and geography interaction with gender
fig=sns.histplot(df,x="Age", hue="Gender", stat="density", kde=True, common_norm=False)#common norm = false=> normalize each histogram independantly
fig.set_xlabel("Age",fontsize=20)
fig.set_ylabel("Density",fontsize=15)
plt.setp(fig.get_legend().get_texts(), fontsize='20')
plt.setp(fig.get_legend().get_title(), fontsize='20')

#Financial variables and gender
sns.catplot(data=df, x="Gender", y="EstimatedSalary", kind="box")
fig, ax=plt.subplots(figsize=[7,5])
sns.countplot(x='NumOfProducts', hue='Gender', data=df, palette='hls').set(xlabel='Number of products contracted by the customer')
G_CrCard = px.histogram(df, x='Gender', color='HasCrCard', barnorm = 'percent', barmode='relative',text_auto=True)
G_CrCard.show() #similar between genders
G_nb_products = px.histogram(df, x='Gender', color='NumOfProducts', barnorm = 'fraction', barmode='relative',text_auto=True).update_yaxes(categoryorder='total ascending')
G_nb_products.show()#similar shares of nb of products,large part of products 1 and 2
sns.catplot(data=df, x="Gender", y="Tenure", kind="box")


#####Geography
#Pie chart : repartition of countries in the dataset
counts = df['Geography'].value_counts()
df2 = pd.DataFrame({'Geography': counts},
                     index = ['France', 'Spain', 'Germany']
                   )
ax = df2.plot.pie(y='Geography', figsize=(5,5), autopct='%1.1f%%')
sns.displot(data=df, x="Age", hue="Geography", kind="kde",common_norm=False) #Age : similar among countries

#Financial variables
sns.displot(data=df, x="Balance", hue="Geography", kind="kde", common_norm=False) #Balance : large concentration in France and Spain around zero and negative values, Germany concentrated in positive
sns.catplot(data=df, x="Geography", y="EstimatedSalary", kind="box")#Estimated Salary : similar among countries
G_nb_products = px.histogram(df, x='Geography', color='NumOfProducts', barnorm = 'fraction', barmode='relative',text_auto=True).update_yaxes(categoryorder='total ascending')
G_nb_products.show()#similar shares of nb of products,large part of products 1 and 2
sns.catplot(data=df, x="Geography", y="Tenure", kind="box")#Tenure : heterogeneous repartition among countries


#Exit
df['Exited'].value_counts()
sns.countplot(x='Exited', data=df, palette='hls')
plt.show()

#Financial variables
sns.displot(data=df, x="CreditScore", hue="Exited", kind="kde",common_norm=False).set(xlabel='Credit Score')
sns.displot(data=df, x="EstimatedSalary", hue="Exited", kind="kde",common_norm=False).set(xlabel='Estimated Salary')
sns.displot(data=df, x="Balance", hue="Exited", kind="kde",common_norm=False).set(xlabel="Customer's bank account's balance")


sns.catplot(data=df, x="Exited", y="CreditScore", kind="box").set(ylabel='Credit Score')
sns.catplot(data=df, x="Exited", y="EstimatedSalary", kind="box").set(ylabel='Estimated Salary')
sns.catplot(data=df, x="Exited", y="Balance", kind="box").set(ylabel='Balance')
sns.catplot(data=df, x="Exited", y="Tenure", kind="box")

sns.countplot(x='NumOfProducts', hue='Exited', data=df, palette='hls')
sns.countplot(x='HasCrCard', hue='Exited', data=df, palette='hls')
sns.countplot(x='IsActiveMember', hue='Exited', data=df, palette='hls')


#socio geograhic factors
sns.countplot(x='Gender', hue='Exited', data=df, palette='hls')
sns.histplot(data=df, x="Geography", hue="Exited", multiple="stack")
sns.displot(data=df, x="Age", hue="Exited", kind="kde",common_norm=False)
sns.catplot(data=df, x="Exited", y="Age", kind="box")
#exited people have on average a higher balance, which may be due to the large number of zero-balance among the non exited persons

df['Balance'].corr(df['EstimatedSalary'])  #positive correlation 12% between salary and balance
df['Balance'].corr(df['CreditScore'])


#####PROBIT MODEL IMPLEMENTATION
import statsmodels as sm
from statsmodels.discrete.discrete_model import Probit

#create dummy variables for categorical variables
data_final=df
data_final=data_final.join(pd.get_dummies(data_final["Geography"],prefix="Geography")).drop("Geography",axis=1)
data_final=data_final.join(pd.get_dummies(data_final["Gender"],prefix="Gender")).drop("Gender",axis=1)


Y = data_final["Exited"]
X = data_final.drop(['Exited','CustomerId', 'Surname'], axis=1)
model = Probit(Y, X)
probit_model = model.fit()
print(probit_model.summary())

#Export to latex
f = open('myreg.tex', 'w')
f.write(probit_model.summary().as_latex())


########################################################################################################
########################################################################################################
########################################################################################################

############################################################################################################################
##########       Implementing a simple machine learning model (Knn model) with a limited set of variables      #############
############################################################################################################################

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


data = here('Input', 'churn_bank.csv')
# importing the dataset a second time in order to use for the machine learning part
df = pd.read_csv(data)

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
plt.savefig('ML_accur_selected_var_model')
plt.title('Knn model accuracy (subset of variables)')

### Selecting the best model among the ones tested with different KNeighborsClassifier
data = [accur1.index(max(accur1)), max(accur1)]
best_model_select = pd.DataFrame([data], columns=["KNeighborsClassifier", "Accuracy"])

##################################################################################################
####### We implement the Knn ML model for a higher range of variables taken into account #########
##################################################################################################


data = here('Input', 'churn_bank.csv')
# importing the dataset a third time in order perform the ML analysis with all the variables contained in the data frame
df = pd.read_csv(data)

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
plt.title('Knn model accuracy (all variables)')
plt.savefig('ML_accur_model_complete')

### Plotting both accuracy graphs in order to compare respective models' performances
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(np.arange(1,80), accur1, accur2);  # Plot some data on the axes.
ax.set_xlabel('Neighbors')
ax.set_ylabel('Accuracy')
fig.align_labels()
plt.title('Comparison of both Knn models accuracies')
plt.savefig('ML_accur_comparison')

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


### generating the ROC curve graph
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




















#Importing the Dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and Data proccessing 

# -> loading data set

sonar_data = pd.read_csv('./sample_data/sonar-data.csv', header=None)

#It show first 4 row and all of the columns
sonar_data.head()

#number of rows and colums (208 rows and 61 colums)
sonar_data.shape

# describe define the summary datas and statistics it is effecint easly to understand
sonar_data.describe()



#it is show us how many data M or R it is count uniq datas
sonar_data[60].value_counts()



# we are going to predict values and check that
sonar_data.groupby(60).mean()

# separating data and labels

X  = sonar_data.drop(columns=60, axis= 1)   # show us all datas without 60th colums axis=0 for rows and axis= 1 for colums 
Y = sonar_data[60]           # show us label datas M or R 208 datas


# traingn and test datas
# xtrain = train the model independent variable, xtest = after training, test the indepented datas(features) 
# ytrain = train the model dependent variable, ytest = after training, test the dependent datas(labels,targets)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1, stratify=Y, random_state=1)

# easy to understandable 208 datas divide to train = 90percent and for test =10percent
print(X.shape , X_train.shape, X_test.shape)
print(X_train , Y_train)

#MODEL TRAINING --> LOGISTIC REGRESSION

model = LogisticRegression()

# training the logistic regression with training data
model.fit(X_train, Y_train)





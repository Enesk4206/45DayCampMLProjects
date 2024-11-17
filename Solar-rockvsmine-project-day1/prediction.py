#Importing the Dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and Data proccessing 

# -> loading data set

sonar_data = pd.read_csv('./Solar-rockvsmine-project-day1/sample_data/sonar-data.csv', header=None)

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
#print(X.shape , X_train.shape, X_test.shape)
# print(X_train , Y_train)

#MODEL TRAINING --> LOGISTIC REGRESSION

model = LogisticRegression()

# training the logistic regression with training data
model.fit(X_train, Y_train)


# MODEL EVALUATION

#accuracy on training data 

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
# print('Accuracy on training model: ', training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
# print('Accuracy on training model: ', test_data_accuracy)

# MAKING PREDICTIVE SYSTEM

input_data = (0.0177,0.0300,0.0288,0.0394,0.0630,0.0526,0.0688,0.0633,0.0624,0.0613,0.1680,0.3476,0.4561,0.5188,0.6308,
              0.7201,0.5153,0.3818,0.2644,0.3345,0.4865,0.6628,0.7389,0.9213,1.0000,0.7750,0.5593,0.6172,0.8635,0.6592,
              0.4770,0.4983,0.3330,0.3076,0.2876,0.2226,0.0794,0.0603,0.1049,0.0606,0.1530,0.0983,0.1643,0.1901,0.1107,
              0.1917,0.1467,0.0392,0.0356,0.0270,0.0168,0.0102,0.0122,0.0044,0.0075,0.0124,0.0099,0.0057,0.0032,0.0019

              )

# changing the input_data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

if(prediction[0] == 'R'):
    print("The object is a Rock")
    
else:
    print("The object is a Mine")
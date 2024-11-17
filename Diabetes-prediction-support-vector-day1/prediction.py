#importing the dependencies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


#DATA COLLECTION & ANALYSIS
#loading the diabetes dataset to a Pandas Frame

diabetes_dataset = pd.read_csv("C:/Users/Enes/Desktop/ML/Diabetes-prediction-support-vector-day1/sample_data/diabetes.csv")

#printing first 5 rows
# print(diabetes_dataset.head())

#data sizes  ->(rows,colums)
# print(diabetes_dataset.shape)


# getting the statistical measures of the data
# print(diabetes_dataset.describe())

#learn more about Outcome label different type counts
# print(diabetes_dataset['Outcome'].value_counts())


#17:43
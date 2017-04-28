# # Preparation

# load required libraries
import pandas as pd
import numpy as pd

# load iris dataset
from sklearn.datasets import load_iris

data = load_iris()

# # Wrangling

# Practising some commands to get some information how the data looksm

# let's find out type
type(data)

# print everything there is
#data
data.feature_names
data.target_names
data.DESCR
data.target
data.data
type(data.target)
type(data.target)

# separate features and target
x = data.data
y = data.target

# # Time to analyze

# split dataset into training and testing sets
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.35)

# let's see the shape of these matrices and vectors
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

# import KNN model
from sklearn.neighbors import KNeighborsClassifier

# instantiate KNN model
knn = KNeighborsClassifier()

# train the model
knn.fit(xtrain, ytrain)

# predict new data with model
prediction = knn.predict(xtest)
print(prediction)


# # Analyze performance
from sklearn.metrics import accuracy_score
print("KNN accuracy score is: ", accuracy_score(ytest, prediction))


# # Model tuning

# let's try different value for K
knn_tuned = KNeighborsClassifier(n_neighbors=7)
knn_tuned.fit(xtrain, ytrain)
prediction_tuned = knn_tuned.predict(xtest)
print("Tuned KNN accuracy score is: ", accuracy_score(ytest, prediction_tuned))
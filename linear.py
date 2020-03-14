# import tensorflow
# import keras

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# reads from the database file and denotes a separator -> ";"
# usually, the separator is a comma (csv stands for "comma separated values")
data = pd.read_csv("student-mat.csv", sep=";")
# prints the first few lines of the data
# print(data.head())

# selects from data the certain columns wanted -> columns are specified using their given names
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# prints the first few lines of the modified/specified data
# print(data.head())

# sets a variable equal to the name of the column that is going to be predicted
predict = "G3"

# creates an array of the data that has dropped the value of our prediction
dataSet = np.array(data.drop([predict], 1))
# creates an array of just the target values
target = np.array(data[predict])

# creates 4 arrays
# dataSet_train = section of dataSet that is used to test and develop a line
# target_train = section of target that is used by dataSet_train to develop a line
# dataSet_test = section of dataSet that is used to test the data and determine accuracy
# target_test = section of target that is used by dataSet_test to determine accuracy
# dataSet_ test and target_test are variables used to test accuracy of models
# explanation: if a computer has seen 100% of the data, it will no longer need to predict
#      because it would have already seen everything and have it in it's memory
#      therefore, the _train variables will have a section of the data that is used to make
#      a model and create a pattern which can then be tested on the other section of the data
# the test_size of 0.1 takes 10% of the data; thus, the other 90% is used to train the AI
dataSet_train, dataSet_test, target_train, target_test = sklearn.model_selection.train_test_split(dataSet, target, test_size = 0.1)

# LINEAR REGRESSION
# the code finds a line equation --> mx + b
# that best fits the points
linear = linear_model.LinearRegression()

# finds a line that best fits with the data (dataSet_Train) and produces
# the target (target_train)
linear.fit(dataSet_train, target_train)

# tells us the accuracy of our line based on the given test data
accuracy = linear.score(dataSet_test, target_test)

# this lets you see the data sets that are used
# dataSet_train has defined percentage of the database to develop a pattern
# dataSet_test has left over percentage of the database to assess the accuracy
# print(dataSet_train)
# print("####################")
# print(dataSet_test)

print(accuracy)

# Saving a model
# You can use this to either save the highest accuracy model or save a model if the testing
# takes a long time.
# creates a file using pickle and dumps your model, in this case it is under the variable linear
with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)

# reads the pickle file
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# constants for the actual linear line
# linear line is in a multi-dimension space
# number of dimensions is equal to the number of columns/variables you give it
# the greater absolute value of coefficients, the greater the weight it has in
# affecting the outcome
print("m : \n", linear.coef_)
print(" b : \n", linear.intercept_)

# predicting values with our line
predictions = linear.predict(dataSet_test)
# print predictions
#for x in range(len(predictions)):
    # print: predicted outcome, data used to determine prediction, and correct answer
    # print(predictions[x], dataSet_test[x], target_test[x])





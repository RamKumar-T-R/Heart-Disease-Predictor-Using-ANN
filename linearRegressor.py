import pandas as pd
import numpy as np
import sklearn
import joblib

# Importing the dataset
dataset = pd.read_csv('D:/\ML_projects/HeartDisease/Dataset/heart_disease_health_indicators_BRFSS2015.csv')
x = dataset.iloc[:, 1: 20].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Serializing StandardScaler object 
joblib.dump(value = sc, filename = 'D:\ML_projects\HeartDisease\standarscaler.pkl')


# Implementing an ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
# Preventing overfeeding in ANN we use Dropout() method
from keras.layers import Dropout

classifier = Sequential()


# Adding input layer and the first hidden layer
classifier.add(Dense(units = 19, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19)) #input_dim=11, because we have 11 indipendnent columns  # 'acitvation' is activation function  # 'relu' is keyword for rectifier activation function  # 'units' is the no. of nodes in the layer # 'input_dim' is mandatory for first hidden(input Layer) and not mandatory for next consequitive layers
classifier.add(Dropout(rate = 0.1)) # to avoid overfeeding we disable (10/100) noumber of nodes in ANN

# Adding second hidden layer
classifier.add(Dense(units = 19, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# Adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # activation function is not rectifier, instead we use sigmoid for probabiliy based functions # Here we will get only 2 categories(0, 1), when dealing with two of more categories we use the activation = 'softmax'


# Compiling / appy stochastic gradient method to our ANNskle
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # 'optimizer' is the algorithm that we are going to use to develope the weights of each nodes(axioms), here we use stochastic gradient algorithm, in stochastic gradient algorithm we have many algorithm in which we are using 'adam' for our purpose


# Fitting classifier to the Training set
classifier.fit(x_train, y_train, batch_size = 1, epochs = 5) # '' is the size of batch after which the weights are get updated   # '' is the number of epoch

# Serializing the model
joblib.dump(value = classifier, filename = 'D:\ML_projects\HeartDisease\classifier.pkl')

# Deserializing the model
classifier = joblib.load('D:\ML_projects\HeartDisease\classifier.pkl')

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.18) # To take away only the two categories 0 or 1


# Single Prediction
predictRes = classifier.predict(sc.transform([[1, 1, 1, 32, 1, 1, 0, 1, 0, 1, 0, 1, 1, 3, 0, 30, 1, 1, 8]]))

# Finding the confusion matrix
from sklearn.metrics import confusion_matrix
Correction = confusion_matrix(y_test, y_pred)

# Finding the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
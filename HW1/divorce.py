import warnings
import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
dataSet = pd.read_csv('./DataSets/divorce.csv', delimiter=';')

# Split the dataset into features and target values
# X contains all the rows except the last, resulting only the features
X = dataSet.iloc[:, :-1].values
# y contains only the last column of the dataSet, resulting only the target values
y = dataSet.iloc[:, -1].values

# Split the dataset into training and testing sets - 25% test, 75% train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define the models to train on the dataSet
perceptron = Perceptron()
naive_bayes = GaussianNB()
decision_tree = DecisionTreeClassifier()
mlp = MLPClassifier()

# Define the neural network model using a linear stack
# linear stack = input of each layer is the output of the previous layer
# relu = rectified linear unit
# sigmoid activation function = takes a real number and outputs a value between 0 and 1
kerasModel = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(54,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the Keras Neural-Network model
# binary crossentropy = loss function commonly used for binary classification problems
# The adam optimizer is an extended version of SGD
# accuracy is the percentage of correctly classified instances
kerasModel.compile(loss='binary_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

# Train the models
perceptron.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    mlp.fit(X_train, y_train)
# mlp.fit(X_train, y_train)

# Train the Keras Neural-Network model
# epochs = no of complete passes throuth the dataset
# batch size = no of samples processed before the model gets updated
kerasModel.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions
perceptron_pred = perceptron.predict(X_test)
naive_bayes_pred = naive_bayes.predict(X_test)
decision_tree_pred = decision_tree.predict(X_test)
mlp_pred = mlp.predict(X_test)
# Evaluate the Keras Neural-Network model
test_loss, test_acc = kerasModel.evaluate(X_test, y_test)
print('Keras Neural-Network Test accuracy:', round(test_acc, 7), '\n')

# Evaluate the models
print('Perceptron:')
print(confusion_matrix(y_test, perceptron_pred))
print(classification_report(y_test, perceptron_pred))

print('Naive Bayes:')
print(confusion_matrix(y_test, naive_bayes_pred))
print(classification_report(y_test, naive_bayes_pred))

print('Decision Tree:')
print(confusion_matrix(y_test, decision_tree_pred))
print(classification_report(y_test, decision_tree_pred))

print('MLP:')
print(confusion_matrix(y_test, mlp_pred))
print(classification_report(y_test, mlp_pred))

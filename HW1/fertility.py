from tensorflow import keras
import matplotlib
import numpy
from sklearn import linear_model as lm
import pandas

dataSet = pandas.read_csv("./DataSets/fertility_Diagnosis.txt", sep=",")

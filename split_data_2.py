# manual nested cross-validation for random forest on a classification dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pandas
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
data = pandas.read_csv("cancer_rnaseq_data_without_first_column.csv") # default header = True
labels = pandas.read_csv("labels_integers.csv") # default header = True

# Make data compatible for converting to tensors
x = np.asarray(data).astype('float32')
y = np.asarray(labels).astype('float32')

# Set counter
counter = 1

for i in range(5):
    # To do: check random state
    outer_x_train, outer_x_test, outer_y_train, outer_y_test = train_test_split(x, y, train_size= 0.8, test_size= 0.2)

    # save training set
    outfile_outer_train = 'planb_outer_train' + str(counter) + '.txt'
    f = open(outfile_outer_train, 'w')
    for line in range(len(outer_y_train)):
        print(outer_y_train[line], outer_x_train[line], file=f)

    # save test set
    outfile_outer_test = 'planb_outer_test' + str(counter) + '.txt'
    f = open(outfile_outer_test, 'w')
    for line in range(len(outer_y_test)):
        print(outer_y_test[line], outer_x_test[line], file=f)

    # update counter
    counter += 1
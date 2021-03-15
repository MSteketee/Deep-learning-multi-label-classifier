# Multi label classification deep learning model voor RNAseq cancer data voor Machine learning project
# Volgens https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

# - hoeveel layers: hoeveelheid RNA expressed is input, daarna wordt dit o.a. (deels) omgezet in eiwitten die vervolgens dingen
# doen voor een cel (hier een groep cellen) wat iets zou kunnen zeggen over welk type kanker. Dus ca. 2 layers (3 in die tutorial, want daar is output ook een dense layer).
#Parameter BRCA COAD KIPAN
#Activation function {Rectifier, Tanh, Maxout}
#Number of hidden layers {2, 3, 4}
#Number of units per layer [10, 200]
#L1 regularization [0.001, 0.1]
#L2 regularization [0.001, 0.1]
#Input dropout ratio [0.001, 0.1]
#Hidden dropout ratios [0.001, 0.1]

# import packages
import pandas as pandas
import numpy as np
import imblearn
from numpy import mean
from numpy import std
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#define the model
def create_model(hidden_layers=1,activation='relu',neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(80, input_dim=20531, activation=activation))
    for i in range(hidden_layers):
        #Add a hidden layer
        model.add(Dense(neurons, activation=activation))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



# load data
data = pandas.read_csv("data.csv", index_col= 0) # default header = True
labels = pandas.read_csv("one_hot-labels.csv", sep = ";", index_col= 0) # default header = True


#print(data)
#print(labels)
# Make data compatible for converting to tensors
data_as_array = np.asarray(data).astype('float32')
labels_as_array = np.asarray(labels).astype('float32')

#print(data_as_array)
#print(labels_as_array)

# OUD: Maak train en test set
#X_train, X_test, y_train, y_test = train_test_split(data_as_array, labels_as_array, test_size=0.20, random_state=33)

# Split 5 time the data into a test and training set for outer CV
cv_outer = KFold(n_splits=5, shuffle=True)


outer_results = list()
for train_ix, test_ix in cv_outer.split(data_as_array):

    # split data
    X_train, X_test = data_as_array[train_ix, :], data_as_array[test_ix, :]
    y_train, y_test = labels_as_array[train_ix], labels_as_array[test_ix]

    cv_inner = KFold(n_splits=5, shuffle=True)
    model = KerasClassifier(build_fn=create_model, verbose=0)

    batch_size = [8,16,32]
    neurons = [30,40,50]
    hidden_layers = [1,2,3]
    epochs = [10,50,100]
    activation = ['relu','tanh','sigmoid']
    param_grid = dict(batch_size=batch_size,neurons=neurons,hidden_layers=hidden_layers,epochs=epochs,activation=activation)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-2, cv=cv_inner)
    resultgridsearch = grid.fit(X_train,y_train)

    best_model = resultgridsearch.best_estimator_

    yhat = best_model.predict(X_test)
    acc = accuracy_score(y_test, yhat)

    outer_results.append(acc)

    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, resultgridsearch.best_score_, resultgridsearch.best_params_))

print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))

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

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(80, input_dim=20531, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# load data
data = pandas.read_csv("C:/Users/Mischa/Downloads/data.csv", index_col= 0) # default header = True
labels = pandas.read_csv("C:/Users/Mischa/Downloads/one_hot-labels.csv", sep = ";", index_col= 0) # default header = True

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

for train_ix, test_ix in cv_outer.split(data_as_array):

    # split data
    X_train, X_test = data_as_array[train_ix, :], data_as_array[test_ix, :]
    y_train, y_test = labels_as_array[train_ix], labels_as_array[test_ix]

    # Balance data set volgens http://glemaitre.github.io/imbalanced-learn/generated/imblearn.over_sampling.RandomOverSampler.html
    # oversample samples < average
    no_samples = np.count_nonzero(y_train, axis = 0)
    average_samples = int(mean(no_samples))
    weights = []
    for i in range(len(no_samples)):
        if no_samples[i] < average_samples:
            weights.append(average_samples)
        else:
           weights.append(no_samples[i])

    ratio_over = {0:weights[0], 1:weights[1], 2:weights[2], 3:weights[3], 4:weights[4]}
    over = SMOTE(sampling_strategy = ratio_over, random_state = 314)
    X_train,y_train = over.fit_resample(X_train,y_train)
    # undersample samples > average
    ratio_under = {0:average_samples, 1:average_samples, 2:average_samples, 3:average_samples, 4:average_samples}
    under = RandomUnderSampler(sampling_strategy = ratio_under, random_state = 314)
    X_train,y_train = under.fit_resample(X_train,y_train)

    # Configure the cross-validation procedure
    cv_inner = LeaveOneOut()

    # Hieronder is voor parameters testen

    # Maak model
    # AANPASSEN NAAR ONS MODEL IPV BASELINE_MODEL
    # model = baseline_model()
    estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=20, verbose=1)

    # kfold = KFold(n_splits=10, shuffle=True) OUD: verwijderen omdat we nu cv = cv_inner i.p.v. cv = kfold gebruiken
    results = cross_val_score(estimator, X_train, y_train, cv=cv_inner)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    #model.fit(X_train, y_train, batch_size = 10, epochs = 1, verbose = 1, validation_split = 0.2)
    #print(X_test)
    #print(model.predict(X_test))
    #score = model.evaluate(X_test, y_test, verbose=1)
    # RESULTS 
    #print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

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

#todo in KerasClassifier ook class balance fixen


# import packages
import pandas as pandas
import numpy as np
from numpy import mean
from numpy import std
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

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


# Maak train en test set
X_train, X_test, y_train, y_test = train_test_split(data_as_array, labels_as_array, test_size=0.20, random_state=33)

# Maak class weights voor class imbalance
label_integers =np.argmax(labels_as_array, axis=1)
class_weights = compute_class_weight('balanced', np.unique(label_integers), label_integers)
d_class_weights = dict(enumerate(class_weights))
print(d_class_weights)
# define baseline model
def baseline_model():
 # create model
 model = Sequential()
 model.add(Dense(50, input_dim=20531, activation='relu'))
 model.add(Dense(20, activation='relu'))
 model.add(Dense(5, activation='softmax'))
 # Compile model
 model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 return model

# Hieronder is voor parameters testen
#estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=20, verbose=1)
#kfold = KFold(n_splits=10, shuffle=True)
#results = cross_val_score(estimator, X_train, y_train, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Test model met uiteindelijke parameters
model = baseline_model()
model.fit(X_train, y_train, batch_size = 10, epochs = 200, verbose = 1, validation_split = 0.2, class_weight = d_class_weights)
print(model.predict(X_test))
score = model.evaluate(X_test, y_test, verbose=1)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


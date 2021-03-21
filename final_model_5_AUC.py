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
from numpy import ravel
from matplotlib import pyplot as plt
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
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from keras.optimizers import Adam
from sklearn.model_selection import LeavePOut
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from numpy import interp
from sklearn.metrics import roc_auc_score
from itertools import cycle

#define the model
def create_model(hidden_layers=1,activation='relu',neurons=1,learning_rate=0.001):
    # create model
    model = Sequential()
    model.add(Dense(80, input_dim=200, activation=activation))
    for i in range(hidden_layers):
        model.add(Dense(neurons,activation=activation))
    model.add(Dense(5,activation="softmax"))
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def apply_pca(data, nr_of_pc):
    if nr_of_pc == 0:
        pca = PCA(.95) # minimal number of components to explain 95% of the variance
    else: pca = PCA(nr_of_pc)

    pca.fit(data)
    #nr_of_pc = pca.n_components_
    result = pca.transform(data)
    return result

def plot_roc_curve(fpr,tpr,roc_auc):
    lw = 2
    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    u = ""
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic plot')
    plt.legend(loc="lower right")
    plt.show()



def RFE_SVM(data, labels, n):
    # returns a list of the n most important features
    selected_features = []
    model = SVR(kernel='linear')
    rfe = RFE(model, n_features_to_select=n, step=100, verbose=0)
    rfe.fit(data, labels)
    for i in range(data.shape[1]):
        if rfe.ranking_[i] == True:
            selected_features.append(data.columns.values[i])
    return selected_features

# load data
data = pandas.read_csv("data.csv", index_col= 0) # default header = True
labels = pandas.read_csv("one_hot-labels.csv", sep = ";", index_col= 0) # default header = True


#print(data)
#print(labels)


# Make data compatible for converting to tensors
data_as_array = np.asarray(data).astype('float32')
labels_as_array = np.asarray(labels).astype('float32')

#print(data_as_array)
# print(labels_as_array)

nr_of_pc = 200
data_as_array = apply_pca(data_as_array, nr_of_pc)

# Split 5 time the data into a test and training set for outer CV
cv_outer = KFold(n_splits=2, shuffle=True)


outer_results = list()
outer_parameters = list()
results_dict = {}
for train_ix, test_ix in cv_outer.split(data_as_array):

    # split data
    X_train, X_test = data_as_array[train_ix, :], data_as_array[test_ix, :]
    y_train, y_test = labels_as_array[train_ix], labels_as_array[test_ix]

    # Balance data set volgens http://glemaitre.github.io/imbalanced-learn/generated/imblearn.over_sampling.RandomOverSampler.html
    # oversample samples < average
    no_samples = np.count_nonzero(y_train, axis=0)
    average_samples = int(mean(no_samples))
    weights = []
    for i in range(len(no_samples)):
        if no_samples[i] < average_samples:
            weights.append(average_samples)
        else:
            weights.append(no_samples[i])

    ratio_over = {0: weights[0], 1: weights[1], 2: weights[2], 3: weights[3], 4: weights[4]}
    over = SMOTE(sampling_strategy=ratio_over, random_state=314)
    X_train, y_train = over.fit_resample(X_train, y_train)

    # undersample samples > average
    ratio_under = {0: average_samples, 1: average_samples, 2: average_samples, 3: average_samples, 4: average_samples}
    under = RandomUnderSampler(sampling_strategy=ratio_under, random_state=314)
    X_train, y_train = under.fit_resample(X_train, y_train)
    cv_inner = KFold(n_splits=2, shuffle=True)
    model = KerasClassifier(build_fn=create_model, batch_size=32, epochs=100, verbose=0)
    learning_rate = [0.001,0.1]
    epochs = [10]
    batch_size = [8]
    neurons = [30]
    hidden_layers = [1, 2]
    activation = ['relu']
    param_grid = dict(learning_rate=learning_rate,epochs=epochs,batch_size=batch_size,neurons=neurons,hidden_layers=hidden_layers, activation=activation)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-2, cv=cv_inner, verbose=1)
    resultgridsearch = grid.fit(X_train,y_train)
    grid_score = resultgridsearch.cv_results_['mean_test_score']
    params = resultgridsearch.cv_results_['params']
    for score,param in zip(grid_score,params):
        results_dict[score] = param

    sorted_acc = sorted(results_dict.keys(), reverse = True)
    for acc in sorted_acc:
        if acc < 0.9:
            final_model_params = results_dict[acc]
            break
    final_model = create_model(hidden_layers = final_model_params["hidden_layers"], activation= final_model_params["activation"],
                               neurons = final_model_params["neurons"], learning_rate = final_model_params["learning_rate"])
    final_model.fit(X_train,y_train,epochs=final_model_params["epochs"],batch_size=final_model_params["batch_size"])

    # gaat vanaf hier nog fout
    n_classes = 5
    y_score = final_model.predict_proba(X_test,batch_size=final_model_params["batch_size"])
    print(y_score)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plot_roc_curve(fpr,tpr,roc_auc)
    print(roc_auc)
    acc = accuracy_score(y_test,y_score) # y_test toegevoegd
    # outer_parameters.append(best_model_params)
    # outer_results.append(acc)
    # print('outer_parameters =', outer_parameters)
    print('outer_results =', outer_results)

print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
print(outer_parameters)

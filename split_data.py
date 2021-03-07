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

# Example: create dataset
#X, y = make_classification(n_samples=1000, n_features=20, random_state=1, n_informative=10, n_redundant=10)

# Load data
data = pandas.read_csv("cancer_rnaseq_data_without_first_column.csv") # default header = True
labels = pandas.read_csv("labels_integers.csv") # default header = True

# Make data compatible for converting to tensors
X = np.asarray(data).astype('float32')
y = np.asarray(labels).astype('float32')

#print("X =", X)
#print("y =", y)

# configure the cross-validation procedure
cv_outer = KFold(n_splits=5, shuffle=True, random_state=1) # I make 5 splits

# Set counter
counter = 1

# enumerate splits
outer_results = list()
for outer_train_ix, outer_test_ix in cv_outer.split(X):

	# split data
	outer_X_train, outer_X_test = X[outer_train_ix, :], X[outer_test_ix, :]
	outer_y_train, outer_y_test = y[outer_train_ix], y[outer_test_ix]

	# save training set
	outfile_outer_train = 'outer_train' + str(counter) + '.txt'
	f = open(outfile_outer_train, 'w')
	for line in range(len(outer_y_train)):
		print(outer_y_train[line], outer_X_train[line], file=f)

	# save test set
	outfile_outer_test = 'outer_test' + str(counter) + '.txt'
	f = open(outfile_outer_test, 'w')
	for line in range(len(outer_y_test)):
		print(outer_y_test[line], outer_X_test[line], file=f)

	# update counter
	counter += 1

	# configure the cross-validation procedure
	cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)

'''    
	# define the model
	model = RandomForestClassifier(random_state=1)
	# define search space
	space = dict()
	space['n_estimators'] = [10, 100, 500]
	space['max_features'] = [2, 4, 6]
	# define search
	search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
	# execute search
	result = search.fit(X_train, y_train)
	# get the best performing model fit on the whole training set
	best_model = result.best_estimator_
	# evaluate model on the hold out dataset
	yhat = best_model.predict(X_test)
	# evaluate the model
	acc = accuracy_score(y_test, yhat)
	# store the result
	outer_results.append(acc)
	# report progress
	print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
'''
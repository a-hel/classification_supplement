import loadfile
import classifiers

import sys

import numpy as np
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

def _find_estimator(X, y, classifier):
	"""Inner loop, find best hyperparameters"""
	estimator = classifier['clf']()
	parameters = classifier['params']
	clf_name = classifier['name']
	print('  Starting grid search for %s...' % clf_name)
	clf = GridSearchCV(estimator=estimator, param_grid=parameters,
    	n_jobs=-1)
	clf.fit(X, y)
	return clf

def _get_labels(y):
	#write nicer
	y = y.fillna('')
	y_e = [True if 'E' in i else False for i in y]
	y_s = [True if 'S' in i else False for i in y]
	y_c = [True if 'C' in i else False for i in y]
	return y_e, y_s, y_c

def nested_cross_val(X, y):
	y_e, y_s, y_c = _get_labels(y)
	y_all = {'E': y_e, 'S': y_s, 'C': y_c}

	for classifier in classifiers.get_classifiers():
		for strain in ('E', 'S', 'C'):
			X_train, X_test, y_train, y_test = cross_validation.train_test_split(
 		 	   X, y_all[strain], test_size=0.33, random_state=42)
			print('Cross-validation for %s' % strain)
			clf = _find_estimator(X_train, y_train, classifier)
			predicted = cross_validation.cross_val_predict(clf, X_test,
				y_test, cv=10)
			pred_rate = (1-np.mean(predicted^y_test))
			print('    Best estimator score: %s' % clf.best_score_)
			print('    Cross-validated prediction rate: %s' % pred_rate)
			yield [classifier['name'], strain, pred_rate, clf.best_estimator_]

def save_buffer(f_buffer, f_name="results.txt"):
	with open("results.txt", "a") as f:
		for line in f_buffer:
			f.write("|".join(str(i) for i in line))
			f.write("\n")

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("\n!   Invalid command: Please provide a file name")
		print("    or, for online datasets, the dataset number.\n")
		sys.exit(0)
	df = loadfile.load_data(sys.argv[1])
	y = df.ix[:,'LABEL']
	X = df.ix[:, '230':]
	
	f_buffer = [["Classifier", "Strain", "Prediction Rate", "Params"]]
	for clf in nested_cross_val(X, y):
		f_buffer.append(clf)
		if len(f_buffer)>=5:
			save_buffer(f_buffer, 'results.txt')
			f_buffer = []
	save_buffer(f_buffer, 'results.txt')

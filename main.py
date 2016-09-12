import loadfile
import classifiers

import numpy as np
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV



  
def _active_wells(cols=('B', 'G'), rows=(2, 11), leading_zero=True):
    """Get the cell designation (e.g. 'A01', 'B2') for the indicated area

    Arguments:
    cols (tuple of chars): The first and last column, e.g. ('A','g').
    rows (tuple of ints): The first and last row, e.g. (2,8).
    leading_zero (bool, default=True): Whether to add a leading zero to
        one-digit numbers

    Returns:
    List with all cell names in the area, both with leading zero and
        without."""

    y = [ord(i.upper())-64 for i in cols]
    col_range = range(min(y)-1, max(y))
    row_range = range(min(rows)-1, max(rows))
    if leading_zero:
        fmt = "{0}{1:02d}"
    else:
        fmt =  "{0}{1}"
    ret_val = []
    for y in col_range:
        for x in row_range:
            ret_val += [fmt.format(chr(y+65), x+1)]
    return ret_val

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
	df = loadfile.load_data("collected_data.csv")#"online")#online
	filter_by_well = df['WELL'].isin(_active_wells())
	df = df[filter_by_well]
	y = df.ix[:,'LABEL']
	X = df.ix[:, '230':]
	
	f_buffer = [["Classifier", "Strain", "Prediction Rate", "Params"]]
	for clf in nested_cross_val(X, y):
		f_buffer.append(clf)
		if len(f_buffer)>=5:
			save_buffer(f_buffer, 'results.txt')
			f_buffer = []
	save_buffer(f_buffer, 'results.txt')

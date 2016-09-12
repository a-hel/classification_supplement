from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from genprog import GP, ANN


def get_classifiers():
	classifiers = [
		{'name': 'SVM',
			'clf': svm.SVC,
			'params': [
				{'C': [1, 10, 100, 1000], 'kernel': ['linear', 'rbf']},
				{'C': [1, 10, 100, 1000], 'degree': [3, 5, 7], 'kernel': ['poly']},
				]
			
		},
		{'name': 'RF',
			'clf': RandomForestClassifier,
			'params': 
				{'n_estimators': [10, 25, 50],
				'criterion': ['gini', 'entropy'],
				'max_features': ['auto', 'log2'],
				'max_depth': [None, 50, 100]},
				
			
		},
		{'name': 'NB',
			'clf': GaussianNB,
			'params': {}
			
		},
		{'name': 'ANN',
			'clf': ANN,
			'params': 
				{'maxEpochs': [ 10, 100], 'nodes': ['default']}
			
		},
		{'name': 'GA',
			'clf': GP,
			'params': 
				{'max_depth': [5,7], 'population_size': [500, 100],
				'generations': [100, 500]}
			
		},

	]
	return classifiers
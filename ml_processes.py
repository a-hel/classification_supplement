# -*- coding: utf-8 -*-

"""
Interface to all machine learning algorithms and pre-processing
"""

from __future__ import print_function

import numpy as np
import random

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import pybrain.datasets as pds
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer

import genprog


def split_set(X, y, training_size=0.7):
    """Split the dataset into a training set and a test set.
    
    Arugments:
    X (2-dimensional container): Dataset
    y (list-like): Labels
    training_size (float, default=0.7): Size of the training
        set
    """

    total_classes = len(set(y))
    n_classes = 1
    set_size = X.shape[0]
    while n_classes < total_classes:
        train_size = int(set_size*(training_size))
        indices = np.arange(set_size)
        random.shuffle(indices)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        X_train = X[train_idx,:]
        y_train = y[train_idx]
        X_test = X[test_idx,:]
        y_test = y[test_idx]
        n_classes = len(set(y_train))
        training_size = training_size*1.1
    return X_train, y_train, X_test, y_test




def pca(X, y, **kwargs):
    """Perform principal component analysis on DataSet X.
    X (DataSet): The dataset on which to apply the function
    kwargs: Keyword arguments to be passed to PCA function.
    Returns all principal components.
    """

    pca = PCA(**kwargs)
    ret_val = pca.fit_transform(X)
    return ret_val, y


def tsne(X, y, **kwargs):
    """Perform t-distributed Stochastic Neighbor Embedding on DataSet X.
    X (DataSet): The dataset on which to apply the function
    kwargs: Keyword arguments to be passed to tsne function.
    Returns t-SNE transformed data.
    """

    model = TSNE(n_components=10, random_state=0, perplexity=20, init="pca")
    ret_val = model.fit_transform(X)
    return ret_val, y

def remove_cp(X, y, comp_remove=1, **kwargs):
    """Remove the first component (which contains outliers) and return the
    back-transformed matrix
    """

    classes = set(y)
    groups = map(tuple, classes)
    unique_groups = set(groups)
    indexes = [[i for i,x in enumerate(groups) if x == unique_group] for 
        unique_group in unique_groups]
    data = X.as_matrix()
    for index in indexes:
        pca = PCA(**kwargs)
        pca.fit(data[index])
        data = pca.transform(data)
    return data.transpose(), y


def SVM(X_train, y_train, X_test, y_test, **kwargs):
    """Support vector machine
    """

    clf = svm.SVC(C=2, kernel='poly', degree=3)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    return prediction, y_test

def ann(X_train, y_train, X_test, y_test, **kwargs):
    """Artificial neural network
    """

    ds = pds.ClassificationDataSet(X_train.shape[1], nb_classes=2,
        class_labels=['T', 'F']) #X.k, X.strains
    for i in range(X_train.shape[0]):
        ds.appendLinked(X_train[i,:], y_train[i])
    # input (size training set, hidden layers, length classification=1)
    nn = buildNetwork(X_train.shape[1], X_train.shape[1]/2, 1, outclass=SoftmaxLayer)
    # trainer: back-propagation
    trainer = BackpropTrainer(nn, ds)
    #trainer.train() # alternative training
    trainer.trainUntilConvergence(maxEpochs=10000) # alternative training
    pre_alloc = np.zeros(X_test.shape[0])
    for i,e in enumerate(X_test):
        pre_alloc[i] = nn.activate(e)
    prediction = np.around(pre_alloc, 0).astype('bool')
    return prediction, y_test

def gp(X_train, y_train, X_test, y_test, **kwargs):
    """Genetic algorithm
    """

    gpr = genprog.GP(**kwargs)
    gpr.fit(X_train, y_train)
    prediction = gpr.predict(X_test)
    return prediction, y_test


def rand_forest(X_train, y_train, X_test, y_test, **kwargs):
    """Random forest
    """

    clf = RandomForestClassifier(n_estimators=25, **kwargs)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    return prediction, y_test

def naive_bayes(X_train, y_train, X_test, y_test, **kwargs):
    """Naive Bayes
    """

    gnb = GaussianNB(**kwargs)
    gnb.fit(X_train, y_train)
    prediction = gnb.predict(X_test)
    return prediction, y_test

if __name__ == "__main__":
    pass
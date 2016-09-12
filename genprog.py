
################################################################################
###                                                                          ###
###  Genetic algorithm interface.                                            ###
###                                                                          ###
################################################################################

from pyevolve import Util
from pyevolve import GTree
from pyevolve import GSimpleGA, Consts, Selectors
import numpy as np


import pybrain.datasets as pds
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer


# If you add new functions, remember to load them to the GA in the GP.fit()
# function.

def gp_add(a, b): return a + b
def gp_sub(a, b): return a - b
def gp_mul(a, b): return a * b
def gp_div(a, b):
    try:
        return a/b
    except ZeroDivisionError:
        return a
def gp_square(a): return a * a
def gp_half(a):   return a*0.5


class GP(object):
    """Object that contains the GP interface, as to be consistant with scikit-
    learn machine learning modules."""

    def __init__(self, max_depth=5, population_size=1000,
        generations=500, **kwargs):
        self.rmse_accum = Util.ErrorAccumulator()
        genome = GTree.GTreeGP(root_node=None)
        genome.setParams(max_depth=max_depth, method="ramped")#max_depth previously at 7
        genome.evaluator.set(self._eval)
        self.genome = genome
        self.ga = False
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth

    def __str__(self):
        return "GP(max_depth='%s', population_size='%s', " +
            "generations='%s')" % (self.max_depth,
            self.population_size, self.generations)

    def get_params(self, deep=False):
        params = {'max_depth': self.max_depth,
            'population_size': self.population_size,
            'generations': self.generations}
        return params

    def set_params(self, **parameters):
        for keys in parameters:
            setattr(self, keys, parameters[keys])
        return self

    def score(self, X, y):
        return np.mean(self.predict(X))

    def fit(self, X, y):
        """ Build predictive model based on data.
        data (pandas DataFrame): Training set
        key (list of booleans): Class prediction.
        """

        self.X = X
        self.y = y
        terminals = ['self.X.iloc[:,{0}]'.format(x) for x
            in range(X.shape[1])]
        ga = GSimpleGA.GSimpleGA(self.genome)
        ga.setParams(gp_function_set = {"gp_add" :2,
                                "gp_sub" :2,
                                "gp_mul":2,
                                "gp_div": 2,
                                "gp_square": 1,
                                "gp_half": 1})
        ga.setParams(gp_terminals=terminals)
        ga.setMinimax(Consts.minimaxType["maximize"])
        ga.selector.set(Selectors.GRouletteWheel)
        ga.setGenerations(self.generations)
        ga.setCrossoverRate(1.0)
        ga.setMutationRate(0.50)
        ga.setPopulationSize(self.population_size)
        ga(freq_stats=100)
        self.ga = ga
        self.best = self.ga.bestIndividual()
        self.win_code = self.best.getCompiledCode()

    def predict(self, X):
        """Predict classification of validation set data.
        data (pandas DataFrame): The validation set.
        """

        self.X = X
        score = eval(self.win_code)
        score = score-score.min()
        score = score/score.max()
        score = np.around(score,0).astype('bool')
        return score

    def _eval(self, genome):
        """Evaluation function for the GA. Might need optimization."""

        code_comp = genome.getCompiledCode()
        self.rmse_accum.reset()
        modification = eval(code_comp)
        modification = modification-modification.min()
        modification = modification/modification.max()
        score = sum(abs(self.y - modification))
        self.rmse_accum   += (score, (modification))
        return score

class ANN(object):
    def __init__(self, nodes='default', maxEpochs=10000, **kwargs):
        self.ds = None
        self.nn = None
        self.nodes = nodes
        self.maxEpochs = maxEpochs

    def __str__(self):
        return "ANN(nodes='%s', maxEpochs='%s')" % (self.nodes,
            self.maxEpochs)

    def get_params(self, deep=False):
        params = {'nodes': self.nodes,
            'maxEpochs': self.maxEpochs}
        return params

    def set_params(self, **parameters):
        for keys in parameters:
            setattr(self, keys, parameters[keys])
        return self

    def fit(self, X, y):
        if self.nodes == 'default':
            self.nodes = int(X.shape[1]/2)
        self.ds = pds.ClassificationDataSet(X.shape[1], nb_classes=2,
            class_labels=['T', 'F']) #X.k, X.strains
        for i in range(X.shape[0]):
            self.ds.appendLinked(X.iloc[i,:], y[i])
        # input (size training set, hidden layers, length classification=1)
        self.nn = buildNetwork(X.shape[1], self.nodes, 1,
             outclass=SoftmaxLayer)
        trainer = BackpropTrainer(self.nn, self.ds)
        #trainer.train() # alternative training
        trainer.trainUntilConvergence(maxEpochs=self.maxEpochs) # alternative training
        
   

    def predict(self, X):
        pre_alloc = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            pre_alloc[i] = self.nn.activate(X.iloc[i,:])
        prediction = np.around(pre_alloc, 0).astype('bool')
        return prediction

    def score(self, X, y):
        prediction = self.predict(X)
        score = 1-(np.mean(prediction^y)) #insert obj_func
        return score


    # trainer: back-propagation
    
    
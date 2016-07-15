
################################################################################
###                                                                          ###
###  Genetic algorithm interface.                                            ###
###                                                                          ###
################################################################################

from pyevolve import Util
from pyevolve import GTree
from pyevolve import GSimpleGA, Consts, Selectors
import numpy as np

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

    def __init__(self, **kwargs):
        self.rmse_accum = Util.ErrorAccumulator()
        genome = GTree.GTreeGP(root_node=None)
        genome.setParams(max_depth=5, method="ramped")#max_depth previously at 7
        genome.evaluator.set(self._eval)
        self.genome = genome
        self.ga = False

    def fit(self, X, y):
        """ Build predictive model based on data.
        data (pandas DataFrame): Training set
        key (list of booleans): Class prediction.
        """

        self.X = X
        self.y = y
        terminals = ['self.X[:,{0}]'.format(x) for x
            in range(X.shape[1])]
        ga = GSimpleGA.GSimpleGA(self.genome)
        ga.setParams(gp_function_set = {"gp_add" :2,
                                "gp_sub" :2,
                                "gp_mul":2,
                                "gp_div": 2,
                                "gp_square": 1,
                                "gp_half": 1})
        ga.setParams(gp_terminals=terminals)
        ga.setMinimax(Consts.minimaxType["minimize"])
        ga.selector.set(Selectors.GRouletteWheel)
        ga.setGenerations(500)
        ga.setCrossoverRate(1.0)
        ga.setMutationRate(0.50)
        ga.setPopulationSize(250)
        ga(freq_stats=10)
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

import numpy as np

from manifolds import MultinomialManifold
from solvers import GradientDescent, ConjugateGradient
from datasets import SyntheticDataSet
from problems import Problem
from evals import Eval
from plots import Plots


    
if __name__ == '__main__':
    
    # pick a dataset
    dataset = SyntheticDataSet(nImg=20, K=10, p=0.2)
    Xin, Xgt, dimGroup = dataset.getData()

    # define the manifold
    manifold = MultinomialManifold(dataset.nImg*dataset.K, dataset.K)

    # pick a solver
    #solver = GradientDescent(tol = 1e-3, learning_rate = 0.1, max_iter = 500)
    solver = ConjugateGradient(tol = 1e-2, max_iter = 500)

    # define initial point
    X0 = manifold.random()

    # define the problem
    problem = Problem(Xin,dimGroup)
    
    # minimize the objective of the problem using solver
    cache, X = solver.Solve(manifold, problem.cost, problem.egrad, X0)
    
    # discretize
    X = np.round(X)

    # evaluate accuracy
    evaluation = Eval(Xin, Xgt, X)
    evaluation.print_accuracy()

    # plot cost
    plots = Plots(cache)
    plots.plot_cost()

    
    
    

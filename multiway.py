import numpy as np
import argparse

from scripts.manifolds import MultinomialManifold
from scripts.solvers import GradientDescent, ConjugateGradient
from scripts.datasets import SyntheticDataSet, WillowDataSet
from scripts.problems import Problem
from scripts.evals import Eval
from scripts.plots import Plots
from scripts.munkres import Discretization

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="Willow",
            help="Dataset option: either 'Willow' or 'synthetic' are supported")
    
    parser.add_argument("--solver", default="cg",
            help="Solver option: either 'cgd' for conjugate gradient or 'gd' for gradient descent")
    
    parser.add_argument("--i", default=500, type=int,
            help="Maximum number of iterations, default to 500")
    
    parser.add_argument("--lr", default=0.05, type=float,
            help="Learning rate for gradient descent, default to 0.05")
    
    parser.add_argument("--tol", default=1e-3, type=float,
            help="Convergence parameter, default to 1e-3.")
    
    parser.add_argument("--n", type=int,
            help="Number of images for creating the synthetic dataset")
    
    parser.add_argument("--k", type=int,
            help="Universe size for creating the synthetic dataset")
    
    parser.add_argument("--o", type=float,
            help="Outlier rate for creating the synthetic dataset")

    parser.add_argument('--plots', dest='plots', action='store_true',
            help="Flag to decide whether to plot the minimization objective over time, default True")

    parser.add_argument('--no-plots', dest='plots', action='store_false',
            help="Use this flag to skip plots")

    parser.set_defaults(plots=True)

    args = parser.parse_args()
    
    if args.dataset == "Willow":
        dataset = WillowDataSet()
    elif args.dataset == "synthetic":
        if args.n is None:
            raise ValueError(f'Unknown number of images for creating synthetic data.')
        if args.k is None:
            raise ValueError(f'Unknown universe size for creating synthetic data.')
        if args.o is None:
            raise ValueError(f'Unknown outlier rate for creating synthetic data.')
        dataset = SyntheticDataSet(nImg=args.n, K=args.k, p=args.o)
    else:
        raise ValueError(f'Unknown dataset option.')

    if args.solver == "cg":
        solver = ConjugateGradient(tol=args.tol, max_iter=args.i)
    elif args.solver == "gd":
        solver = GradientDescent(tol=args.tol, learning_rate=args.lr, max_iter=args.i)
    else:
        raise ValueError(f'Unknown solver...')

    # get the data: input correspondences, groundtruth correspondences
    # and number of features in each image
    Xin, Xgt, dimGroup = dataset.getData()

    # define the manifold
    manifold = MultinomialManifold(dimGroup.sum(), dataset.K)

    # define initial point
    X0 = manifold.random()

    # define the problem
    problem = Problem(Xin, dimGroup)
    
    # minimize the objective of the problem using solver
    cache, X = solver.Solve(manifold, problem.cost, problem.egrad, X0)

    # discretize low-rank factors
    discretization = Discretization(X, dimGroup)
    X = discretization.Stochastic2Permutation()

    # evaluate accuracy
    evaluation = Eval(Xin, Xgt, X)
    evaluation.print_accuracy()

    # plot cost over time
    if args.plots:
        plots = Plots(cache)
        plots.plot_cost()

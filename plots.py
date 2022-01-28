import numpy as np
import matplotlib.pyplot as plt

class Plots():

    """Class that implements plots regarding the optimization method 
    given the info stored during optimization.""" 
    
    def __init__(self, cache=None):
        
        if cache is None:
            raise ValueError('No cache found.')
        
        if len(cache) == 0:
            raise ValueError('Cache is empty.')

        # cache.shape  should be (nIter, 2)
        if len(cache[0]) != 2:
            raise ValueError('Cache provided has incorrect shape.')

        self.cache = cache

    def plot_cost(self):

        fig, ax = plt.subplots()
        ax.semilogy([c[0]-self.cache[-1][0] for c in self.cache])
        ax.grid()
        plt.title('Minimization Objective over Time')
        plt.ylabel('f(X)-p*')
        plt.xlabel('Number of Iterations')
        plt.show()

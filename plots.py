import numpy as np
import matplotlib.pyplot as plt


class Plots():

    # class that implements plots regarding the optimization method
    # given the info stored during optimization
    # cache.shape = (nIter, 2)
    
    def __init__(self, cache):

        self.cache = cache

    def plot_cost(self):

        fig, ax = plt.subplots()
        ax.semilogy([c[0]-self.cache[-1][0] for c in self.cache])
        ax.grid()
        plt.title('Minimization Objective over Time')
        plt.ylabel('f(X)-p*')
        plt.xlabel('Number of Iterations')
        plt.show()

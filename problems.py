import numpy as np
from numpy.linalg import norm

class Problem():

    """Class that implements the cost function and its Euclidean gradient."""

    def __init__(self, Xin, dimGroup, lam=1):

        # noisy input matrix
        self.Xin = Xin

        # number of observed features per image
        self.dimGroup = dimGroup

        # array of indices based on the number of observed features per image
        self.ind = np.insert(self.dimGroup, 0, 0).cumsum()

        # regularization parameter
        self.lam = lam

    def cost(self, X):

        """Compute objective given point X."""

        # first, compute the low-rank factorization penalty
        XTX = np.dot(X.T, X)
        cost = self.Xin.sum() + norm(XTX, 'fro')**2 -2*np.trace(np.dot(X.T, self.Xin.dot(X))) 
        
        # then add regularizers to the objective
        for i in range(1,len(self.dimGroup)+1):
            Xi = X[self.ind[i-1]:self.ind[i],:]
            cost = cost + self.lam*norm( np.eye(Xi.shape[0])-np.dot(Xi,Xi.T) , 'fro')**2
        
        return .25*cost

    def egrad(self, X):

        """Compute Euclidean gradient of objective given point X."""
        
        # first, compute the low-rank factorization penalty gradient
        egrad = -self.Xin.dot(X) + np.dot(X, np.dot(X.T, X))

        # then add gradients of regularizers
        for i in range(1,len(self.dimGroup)+1):
            idx = np.arange(self.ind[i-1], self.ind[i], dtype=int)
            Xi = X[idx,:]
            egrad[idx,:] = egrad[idx,:] - self.lam * (Xi - np.dot(Xi, np.dot(Xi.T, Xi)))

        return egrad

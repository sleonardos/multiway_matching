import numpy as np
from numpy.linalg import norm

class Problem():

    def __init__(self, Xin, dimGroup):

        self.Xin = Xin
        self.dimGroup = dimGroup
        self.csdimGroup = self.dimGroup.cumsum()

    def cost(self, X):

        cost = norm(self.Xin-np.dot(X,X.T),'fro')**2

        Xi = X[:self.csdimGroup[0],:]
        cost = cost + .25*norm( np.eye(Xi.shape[0])-np.dot(Xi,Xi.T), 'fro')**2
        for i in range(1,len(self.dimGroup)):
            Xi = X[self.csdimGroup[i-1]:self.csdimGroup[i],:]
            cost = cost + norm( np.eye(Xi.shape[0])-np.dot(Xi,Xi.T) , 'fro')**2
        
        return .25*cost


    def egrad(self, X):

        egrad = - np.dot(self.Xin,X) + np.dot(X,np.dot(X.T,X))

        idx = np.arange(0,self.csdimGroup[0])
        Xi = X[idx,:]
        egrad[idx,:] = egrad[idx,:] - Xi + np.dot(Xi,np.dot(Xi.T,Xi))
        for i in range(1,len(self.dimGroup)):
            idx = np.arange(self.csdimGroup[i-1], self.csdimGroup[i])
            Xi = X[idx,:]
            egrad[idx,:] = egrad[idx,:] - Xi + np.dot(Xi,np.dot(Xi.T,Xi))
        
        return egrad

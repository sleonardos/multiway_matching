import numpy as np
from numpy.linalg import norm

eps = 1e-10

class MultinomialManifold():

    def __init__(self,N,K):
        
        # embedding space NxK
        self.N = N
        self.K = K

    def random(self):
        
        X = np.random.rand(self.N, self.K)
        return X / X.sum(axis=1,keepdims = True)
    
    def metric(self, X, U, V):
        
        X = np.maximum(eps, X)
        return np.sum((U*V)/X)

    def norm(self, X, U):

        return self.metric(X, U, U)
    
    def checkTangent(self, X, U):
     
        return  norm(U-self.projection(X,U), 'fro') < eps
        
    def projection(self, X, U):

        return U-U.sum(axis = 1, keepdims = True)*X
    
    def randomTangent(self, X):

        U = np.random.randn(self.N, self.K)
        return self.projection(X,U)

    def randomDirection(self, X):

        U = np.random.randn(self.N, self.K)
        U = self.projection(X, U)
        return U/self.norm(X, U)
    
    def retraction(self, X, U):

        #X = np.maximum(eps, X)
        Y = X*np.exp(U/X)
        Y = Y/Y.sum(axis=1, keepdims = True)
        return np.maximum(eps, Y)

    
    def transport(self, X, Y, V):

        return self.projection(Y, V)
    
    def egrad2rgrad(self, X, egrad):

        U = egrad(X)*X
        return self.projection(X, U)

    def ehess2rhess(self, X, egrad, ehess, U):

        # returns Hess f(X)[U]
        G = egrad(X) 
        V = ehess(X,U)*X + 0.5*(G - np.sum(G*X,axis=1,keepdims=True))*U
        return self.projection(X, V)

if __name__ == '__main__':

    N, K = 1, 3
    manifold = MultinomialManifold(N,K)
    X = manifold.random()
    U = manifold.randomDirection(X)
    print(X)
    print(U)
    

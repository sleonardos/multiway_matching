import numpy as np
from numpy.linalg import norm

eps = 1e-10

class MultinomialManifold():

    # class that implements all the ingeredients for
    # optimization on the manifold of NxK stochastic matrices
    # based on the Riemannian manifold structure when endowed with
    # the Fisher information metric
    def __init__(self,N,K):
        
        # embedding space NxK
        self.N = N
        self.K = K

    def random(self):

        # generate a random point on the manifold
        X = np.random.rand(self.N, self.K)
        return X / X.sum(axis=1,keepdims = True)
    
    def metric(self, X, U, V):

        # Riemannian metric <U,V>_X  
        X = np.maximum(eps, X)
        return np.sum((U*V)/X)

    def norm(self, X, U):

        # get norm ||U||_X = <U,U>_X^(1/2)
        return np.sqrt(self.metric(X, U, U))
    
    def checkTangent(self, X, U):

        # check if U is a tanget vector at X
        return  norm(U-self.projection(X,U), 'fro') < eps
        
    def projection(self, X, U):

        # project a vector U to the tangent space at X
        return U-U.sum(axis = 1, keepdims = True)*X
    
    def randomTangent(self, X):

        # generate a random tangent space at X
        U = np.random.randn(self.N, self.K)
        return self.projection(X,U)

    def randomDirection(self, X):

        # generate a random direction on the tangent space at X
        U = np.random.randn(self.N, self.K)
        U = self.projection(X, U)
        return U/self.norm(X, U)

    def exp_normalize(self, X):

        # softmax function
        Y = np.exp(X - np.amax(X, axis=1,keepdims=True) )
        return Y / Y.sum(axis=1, keepdims = True)
        
    def retraction(self, X, U):

        # compute the retraction by computing a softmax function
        Z = np.log(np.maximum(X,eps)) +  U/X
        Y = self.exp_normalize(Z)
        return np.maximum(eps, Y)

    def transport(self, X, Y, V):

        # vector transport of V from the tangent space of X to the tangent
        # space at Y, compatible with rectraction
        return self.projection(Y, V)
    
    def egrad2rgrad(self, X, egrad):

        # compute Riemannian gradient from Euclidean gradient
        U = egrad(X)*X
        return self.projection(X, U)

    def ehess2rhess(self, X, egrad, ehess, U):

        # returns Riemannian Hess f(X)[U] from Euclidean Hessian and gradient
        G = egrad(X) 
        V = ehess(X,U)*X + 0.5*(G - np.sum(G*X,axis=1,keepdims=True))*U
        return self.projection(X, V)

if __name__ == '__main__':

    # sample testcase
    N, K = 1, 3
    manifold = MultinomialManifold(N,K)
    
    X = manifold.random()
    U = manifold.randomDirection(X)
    
    print(X)
    print(U)

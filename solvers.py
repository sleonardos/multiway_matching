import numpy as np
import matplotlib.pyplot as plt
from manifolds import MultinomialManifold

class GradientDescent():

    # class that implements gradient descent with constant stepsize
    
    def __init__(self, learning_rate = 0.1, max_iter = 1000, tol = 1e-4, pert = 1e-6):

        # learning rate
        self.learning_rate = learning_rate

        # maximum number of iterations
        self.max_iter = max_iter

        # gradient norm tolerance, stops if norm of gradient below tol
        self.tol = tol

        # a parameter for random perturbation to avoid saddle points
        self.pert = pert

    def Solve(self, manifold, cost, egrad, X):

        # compute Riemannian gradient from Euclidean gradient
        grad = manifold.egrad2rgrad(X, egrad)

        # compute cost, gradient norm and cache them
        grad_norm, costX = manifold.norm(X, grad), cost(X)
        cache = [(costX,grad_norm)]
        
        for i in range(self.max_iter):

            # compute Riemannian gradient from Euclidean gradient
            grad = manifold.egrad2rgrad(X, egrad)

            # update in the direction of negative gradient and project back to the manifold
            X = manifold.retraction(X, -self.learning_rate*grad)
            
            # perturb to avoid saddle points faster
            #X = manifold.retraction(X, self.pert*manifold.randomDirection(X))

            # compute cost, gradient norm and cache them
            costX, grad_norm  = cost(X), manifold.norm(X,grad)
            cache.append((costX,grad_norm))

            if i==0:
                print("=================== Gradient Descent =======================")
                
            if i%10==0:
                print("Iteration number  %.0f with cost %.4f" % (i,costX))
            
            if grad_norm < self.tol:
                break
            
        return cache, X

class ConjugateGradient():

    # class that implements conjugate gradient with line-search

    def __init__(self, max_iter = 1000, tol = 1e-4, pert = 1e-6):

        # maximum number of iterations
        self.max_iter = max_iter

        # gradient norm tolerance, stops if norm of gradient below tol
        self.tol = tol

        # a parameter for random perturbation to avoid saddle points
        self.pert = pert

        
    def LineSearch(self, manifold, cost, X, eta, grad):

        # line search of cost along direction eta
        # based on Armijo rule (backtracking)
        
        alpha, sigma, b, max_cost_evals = 1, 0.5, 0.5, 25
        costX, cost_evals = cost(X), 1

        # computer inner product between gradient and eta
        df = manifold.metric(X, grad, eta)
        
        Y = manifold.retraction(X,alpha*eta)
        # while not sufficient decrease, reduce stepsize by factor 1/b
        while  cost(Y) > costX + alpha*sigma*df:

                # reduce stepszie
                alpha *= b

                # compute new point 
                Y = manifold.retraction(X,alpha*eta)

                # increase number of cost evaluations and check if max number exceeded
                cost_evals+=1
                if cost_evals>=max_cost_evals:
                    return 0
                
        return alpha
        
    def Solve(self, manifold, cost, egrad, X):

        # compute Riemannian gradient from Euclidean gradient
        grad = manifold.egrad2rgrad(X, egrad)

        # compute cost, gradient norm and cache them
        grad_norm, costX = manifold.norm(X, grad), cost(X)
        cache = [(costX,grad_norm)]

        # set initial direction equal to negative gradient
        eta = -grad
        
        for i in range(self.max_iter):

            # check if it eta is a descent direction
            # If not reset it to minus gradient
            if manifold.metric(X, eta, grad) >= 0 :
                eta = -grad

            # line search for step-size
            alpha = self.LineSearch(manifold, cost, X, eta, grad)

            # reset to gradient direction if not reduction
            if alpha==0:
                eta=-grad

            # update estimate and maintain previous
            X, X_old  = manifold.retraction(X, alpha*eta), X
            
            # perturb to avoid saddle points
            X = manifold.retraction(X, self.pert*manifold.randomDirection(X))

            # update gradient and maintain previous
            grad, grad_old = manifold.egrad2rgrad(X, egrad), grad
            
            # Fletcher - Reeves beta
            #beta = manifold.norm(X, grad) / manifold.norm(X_old, grad_old)**2

            # Polak-Ribiere beta
            beta = manifold.metric(X, grad, grad- manifold.transport(X_old, X, grad_old) ) / manifold.norm(X_old, grad_old)**2
            beta = max(0,beta)
    
            # new direction as linear comb of negative gradient and transported old direction
            eta = -grad + beta*manifold.transport(X_old, X, eta)
            
            # compute cost, gradient norm and cache them
            costX, grad_norm  = cost(X), manifold.norm(X,grad)
            cache.append((costX,grad_norm))

            if i==0:
                print("=================== Conjugate Gradient =======================")
                
            if i%10==0:
                print("Iteration number  %.0f with cost %.4f" % (i, costX))
            
            if grad_norm < self.tol:
                break
            
        return cache, X
            
if __name__ == '__main__':
    
    # sample testcase with a simple objective f(X) = (1/4)||I-X*X.T||_F^2
    def cost(X):
        XtX = np.dot(X.T,X)
        return .25*(X.shape[0] - 2*np.trace(XtX) + np.trace(np.dot(XtX, XtX)))

    def egrad(X):
        return (np.dot(X,np.dot(X.T,X)) - X)

    # define the manifold
    N, K = 10, 20
    manifold = MultinomialManifold(N, K)

    # pick a solver
    #solver = GradientDescent()
    solver = ConjugateGradient()

    # define initial point
    X0 = manifold.random()

    # minimize the objective using solver
    cache, X = solver.Solve(manifold, cost, egrad, X0)

    # plot cost
    fig, ax = plt.subplots()
    ax.semilogy([c[0]-cache[-1][0] for c in cache])
    ax.grid()
    plt.title('Minimization Objective over Time')
    plt.ylabel('f(x)-p*')
    plt.xlabel('Number of Iterations')
    plt.show()

    # plot gradient norm
    fig, ax = plt.subplots()
    ax.semilogy([c[1] for c in cache])
    ax.grid()
    plt.title('Gradient Norm over Time')
    plt.ylabel('Gradient Norm')
    plt.xlabel('Number of Iterations')
    plt.show()

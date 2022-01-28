import numpy as np
from scipy.optimize import linear_sum_assignment

class Discretization():

    """Class to implement projection of each stochastic submatrix
       of the low-rank factor X to the set of (partial) permutation matrices.
    """

    def __init__(self, X, dimGroup):

        # low-rank factor matrix
        self.X = X

        # dimension of each block
        self.dimGroup = dimGroup

    
    def Stochastic2Permutation(self):

        # project each stochastic submatrix of X
        # to the set of (partial) permutation matrices
        # using the Hungarian algorithm
        
        idx1, idx2 = 0, self.dimGroup[0]
        for iImg in range(self.dimGroup.shape[0]):

            cost = self.X[idx1:idx2,:]

            # solve the assignment problem
            # ATTN: scipy assumes minimization
            row_ind, col_ind = linear_sum_assignment(-cost)

            # modify the matrix X
            self.X[idx1:idx2,:] = 0
            self.X[idx1+row_ind,col_ind] = 1

            # update indices
            if iImg<self.dimGroup.shape[0]-1:
                idx1 = idx1 + self.dimGroup[iImg]
                idx2 = idx1 + self.dimGroup[iImg+1]

        return self.X

if __name__ == '__main__':

    # Sample testcase
    X = np.array([[0.6, 0.2, 0.2],
                  [0.2, 0.3, 0.5],
                  [0.1, 0.4, 0.5],
                  [0.9, 0.1, 0.0],
                  [0.1, 0.8, 0.1]])
    
    # the discretized X should be equal to
    Xd = np.array([[1., 0., 0.],
                   [0., 0., 1.],
                   [0., 0., 1.],
                   [1., 0., 0.],
                   [0., 1., 0.]])
    
    dimGroup = np.array([2, 3])

    X = Discretization(X, dimGroup).Stochastic2Permutation()

    assert np.all(np.equal(X, Xd))

import numpy as np

class Eval():
    """Class to evaluate performance of multiway matching.

    Compares initial correspondence accuracy to the final accuracy
    obtained by multiway matching. 
    """

    def __init__(self, Xin=None, Xgt=None, X=None):
        
        if Xin is None:
            raise ValueError('Input correspondences not provided.')
        
        if Xgt is None:
            raise ValueError('No groundtruth provided.')
        
        if X is None:
            raise ValueError('No low rank solution provided.')

        # input correspondences
        self.Xin = Xin

        # groundtruth correspondences
        self.Xgt = Xgt

        # recovered low rank factor
        self.X = X

    def accuracy(self, Xp, Xgt):

        return 100*(Xgt.multiply(Xp).sum()/Xgt.sum())

    def print_accuracy(self):

        self.init_accuracy = self.accuracy(self.Xin, self.Xgt)
        self.accuracy = self.accuracy(self.X.dot(self.X.T), self.Xgt)

        print('Initial accuracy: %.2f%%' % self.init_accuracy)
        print('Accuracy after optimization: %.2f%%' % self.accuracy)

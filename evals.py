import numpy as np

class Eval():

    # class to evaluate performance of multiway matching
    # compares initial correspondence accuracy with accuracy
    # after multiway matching
    
    def __init__(self, Xin, Xgt, X):

        # input correspondences
        self.Xin = Xin

        # groundtruth correspondences
        self.Xgt = Xgt

        # recovered low rank factor
        self.X   = X

    def print_accuracy(self):

        init_accuracy = 100*self.Xgt.multiply(self.Xin).sum()/self.Xgt.sum()
        
        accuracy = 100*np.sum(self.Xgt.multiply(np.dot(self.X,self.X.T)))/self.Xgt.sum()

        print("==========================================")
        
        print('Initial accuracy: %.2f %%' % init_accuracy)
        
        print('Accuracy after optimization: %.2f %%' % accuracy)


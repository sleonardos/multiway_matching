import numpy as np

class Eval():

    def __init__(self, Xin, Xgt, X):

        self.Xin = Xin
        self.Xgt = Xgt
        self.X   = X

    def print_accuracy(self):

        init_accuracy = 100*np.sum(self.Xin*self.Xgt)/np.sum(self.Xgt)
        
        accuracy = 100*np.sum(np.dot(self.X,self.X.T)*self.Xgt)/np.sum(self.Xgt)

        print("==========================================")
        
        print('Initial accuracy: %.2f %%' % init_accuracy)
        
        print('Accuracy after optimization: %.2f %%' % accuracy)


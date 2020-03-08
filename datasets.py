import numpy as np

class SyntheticDataSet():

    def __init__(self, nImg=10, K=10, p=0.2):

        # number of images 
        self.nImg = nImg
        
        # size of universe
        self.K = K

        # outlier percentage
        self.p = p

    def getData(self):
        
        Xin = np.eye(self.nImg*self.K)
        
        Xgt = np.tile(np.eye(self.K), (self.nImg,self.nImg))
        
        dimGroup = self.K*np.ones((self.nImg,), dtype=int)
        
        for i in range(self.nImg):
            for j in range(i):
                
                Xij = np.eye(self.K)
                
                P = np.random.permutation(self.K)[:int(self.K*self.p)]
                
                Xij[P,P], Xij[P,np.roll(P,1)] = 0, 1

                Xin[i*self.K:(i+1)*self.K,j*self.K:(j+1)*self.K] = Xij
                Xin[j*self.K:(j+1)*self.K, i*self.K:(i+1)*self.K] = Xij.T

        return Xin, Xgt, dimGroup


if __name__ == '__main__':

    dataset = SyntheticDataSet(nImg=5, K=10, p=0.2)
    Xin, Xgt, dimGroup = dataset.getData()

    print(Xin)

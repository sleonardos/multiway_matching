import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix



class SyntheticDataSet():

    # class to generate synthetic dataset
    # To generate a synthetic dataset we need the following:
    # nImg: number of images
    # K: size of universe or equivalently rank of matrix
    # p: percentage of false correspondences/outliers
    
    def __init__(self, nImg=10, K=10, p=0.2):

        # number of images 
        self.nImg = nImg
        
        # size of universe, or equivalently
        # rank of the matrix of pairwise correspondences
        self.K = K

        # outlier percentage
        self.p = p


    def getData(self):

        # initialize input matrix
        Xin = np.eye(self.nImg*self.K)

        # groundtruth correspondences
        Xgt = csr_matrix(np.tile(np.eye(self.K), (self.nImg,self.nImg)))

        # number of features observed in each view
        dimGroup = self.K*np.ones((self.nImg,), dtype=int)
        
        for i in range(self.nImg):
            for j in range(i):

                # set equal to grountruth
                Xij = np.eye(self.K)

                # generate p*K outliers
                P = np.random.permutation(self.K)[:int(self.K*self.p)]
                
                Xij[P,P], Xij[P,np.roll(P,1)] = 0, 1
                Xin[i*self.K:(i+1)*self.K,j*self.K:(j+1)*self.K] = Xij
                Xin[j*self.K:(j+1)*self.K, i*self.K:(i+1)*self.K] = Xij.T

        return csr_matrix(Xin), Xgt, dimGroup




class WillowDataSet():

    # class to implement data loading for WILLOW object class datasets
    
    def __init__(self, name = 'Motorbikes', nImg = 40, K=10):

        # name of files where data are stored
        self.filename = 'data/Willow' + name + '.mat'

        # number of images
        self.nImg = nImg

        # size of universe, or equivalently
        # rank of the matrix of pairwise correspondences
        self.K = K


    def getData(self): 

        # load file
        data_dict = loadmat(self.filename)

        # extract matrices from dictionary
        Xin = csr_matrix(data_dict['X_in'].toarray())
        Xgt = csr_matrix(data_dict['X_gt'].toarray())
        dimGroup = data_dict['dimGroup'][:,0]
        
        return Xin, Xgt, dimGroup


if __name__ == '__main__':

    #dataset = SyntheticDataSet(nImg=5, K=10, p=0.2)
    dataset = WillowDataSet()
    Xin, Xgt, dimGroup = dataset.getData()

    print(dimGroup)
    print(Xin)
    print(Xgt)

from .abstract_embedding import AbstractEmbedding
import matplotlib.pyplot as plt
import numpy as np

class FactorizationEmbedding(AbstractEmbedding):
    """    
        Abstract class that provides the blueprint for the development of ebedding methods based on matrix factorization
        
    """
    def __init__(self, d, network):
        super().__init__(d, network)
        self.matrix = None
        self.vals = None
        self.vecs = None
        
    def plot_eigenvalues(self, figsize = [8, 8]):
        fig = plt.figure(figsize=figsize)
        plt.scatter(self.vals.real,self.vals.imag)
        limit=np.max(np.ceil(np.absolute(self.vals)))
        plt.xlim((-limit,limit))
        plt.ylim((-limit,limit))
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.title("Matrix Eigenvalues")
        plt.show()       
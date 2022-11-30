import scipy as sp
from .factorization_embedding import FactorizationEmbedding

# https://scikit-network.readthedocs.io/en/latest/_modules/sknetwork/embedding/spectral.html#Spectral 
class LaplacianEigenmap(FactorizationEmbedding):
    def __init__(self, d, network): #, weight = True
        assert check_symmetric(network.adjacency_matrix.todense()), "Network is not symmetric" # hence we are not usign T mat... 
        assert d+1<= network.adjacency_matrix.shape[0], "can't compute embedding with number of dimensions larger than the number of nodes"
        super().__init__(d, network) # in here dself.d = d+1

        self.matrix = sp.sparse.csgraph.laplacian(self.adjacency_matrix, normed=True)
        
    def compute_embedding(self, rescale_eigenvectors = False):
        # vals, vecs =  sp.linalg.eig(self.matrix) #.todense()
        vals, vecs = sp.sparse.linalg.eigs(self.matrix, k = self.d+1, which="SM", return_eigenvectors=True)
    
        #order vals and vecs
        idx = vals.argsort() 
        self.vals = vals[idx]
        self.vecs = vecs[:,idx].real
        
        # remove zero
        self.vals = vals[1:]
        self.vecs = vecs[:,1:].real
        # rescale contributions of eigenvectors
        if rescale_eigenvectors:
            self.vecs = self.vals.real * self.vecs

        self.embedding = {}
        for v in self.nodes:
            if isinstance(v, str):
                v = (v,)
            self.embedding[v] = np.array(self.vecs[self.node_to_index[v],:self.d])

    def __str__(self):
        return "LaplacianEigenmap"
    def attributes_dictionary(self):
        return {
            "method":str(self),
            "d":self.d, 
            "order":self.network.order,
            "directed":self.network.directed
            }

import numpy as np
def check_symmetric(a, rtol=1e-05, atol=1e-05):
    if isinstance(a,sp.sparse.csr.csr_matrix):
        a = a.todense()
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

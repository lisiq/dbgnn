from sklearn.utils.extmath import randomized_svd
from .factorization_embedding import FactorizationEmbedding
import scipy as sp
import numpy as np
class SVD_embedding(FactorizationEmbedding): # Can be used for HONEM
    def __init__(self, d, network, matrix, weight = True, method_name = None):
        assert d<= max(matrix.shape), "can't compute embedding with number of dimensions larger than the input matrix"
        super().__init__(d,network)
        self.d = d
        self.weight = weight
        self.matrix = matrix 
        # if method_name is not None:
        #     self.method_name = method_name
        # else:
        #     self.method_name = "SVD_embedding"
        self.method_name = "HONEM"


    def compute_embedding(self, maxiter = 1e6):
        # being performed through SVD we have a different set of objects
        # U: left singular vector
        # Vh: right singular vector
        # s: singular values
        U, s, Vh = sp.sparse.linalg.svds(
            self.matrix,
            k=self.d,
            maxiter = maxiter,
            which='LM',
            return_singular_vectors=True
            ) # cause smaller is zero... (should I keep it when not zero?When not zero no strongly connected)
        self.basis = Vh
        self.vals = s
        self.vecs = U
        # rescaling the eigenvectors (avoids goving all the same weight)
        U = np.sqrt(s)*U# this as done in HONEM, are there alternatives?

        self.embedding = {}
        for v in self.network.nodes:
            # if isinstance(v, str):
            #     v = (v,)
            self.embedding[v] = U[self.node_to_index[v],:] 



    def compute_truncatedSVD_embedding(self):
        """
            https://stackoverflow.com/questions/31523575/get-u-sigma-v-matrix-from-truncated-svd-in-scikit-learn
        """

        U, s, Vh = randomized_svd(self.matrix.todense(),
                                    n_components=self.d,
                                    n_iter=int(1e3),
                                    random_state=None)
        self.basis = Vh
        self.vals = s
        self.vecs = U
        U = np.sqrt(s)*U

        self.embedding = {}
        for v in self.nodes:
            self.embedding[v] = U[self.node_to_index[v],:] #vecs[n.nodes.index[v],:k]


    def plot_embedding(self,edges, colors = None,plot_names = False):
        # assert nodes in input network are ok
        self.edges = edges
        super(SVD_embedding, self).plot_embedding(colors = colors, plot_names = plot_names)

    def __str__(self):
        return self.method_name
    def attributes_dictionary(self):
        return {
            "method":str(self),
            "d":self.d, 
            "order":self.network.order,
            "directed":self.network.directed}

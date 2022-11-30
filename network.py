import numpy as np 
import scipy as sp
def check_symmetric(a, rtol=1e-05, atol=1e-05):
    if isinstance(a,sp.sparse.csr.csr_matrix):
        a = a.todense()
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

class Network_light():
    def __init__(self, adjacency_matrix, node_to_index):
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1], "Adjacency matrix has to be a square matrix"

        self.adjacency_matrix = adjacency_matrix
        self.directed = not check_symmetric(adjacency_matrix) # creates dense matrix - issues when loading large hons
        self.weighted = len(set(self.adjacency_matrix.data)) != 1
        # self.transition_matrix = self.adjacency_matrix /self.adjacency_matrix.sum(axis = 1) #remove_nan_from_csr(sp.sparse.csr_matrix(self.adjacency_matrix /self.adjacency_matrix.sum(axis = 1)))
        #
        self.node_to_index = node_to_index
        self.index_to_node = {v:k for k,v in self.node_to_index.items()}
        self.nodes = set(self.node_to_index.keys())
        self.n_nodes = len(self.nodes)
        self.start_probabilities = None
        self.end_probabilities = None
        self.order = 1
        #
        self.outdegree = {}
        for ix, node in self.index_to_node.items():
            self.outdegree[node] = self.adjacency_matrix[ix,:].sum()

        #
        # edges with weight not implemented here
        self.edges = self.__get_edges()
        #
        # self.dynamics = dynamic # what should this be?
        self.__successors = dict()

    def __get_edges(self):
        # if self.edges: # TODOs: 
        #     return self.edges
        us, vs = self.adjacency_matrix.nonzero()
        return [(self.index_to_node[us[i]],self.index_to_node[vs[i]]) for i in range(len(us))]

    def __get_successors(self, node_id): # successors, actually...
        """
            returns the indices of the nonzero column, as well as their values (ordered)
        """
        row_ix = self.node_to_index[node_id]
        column_indices = self.adjacency_matrix.indices[
            self.adjacency_matrix.indptr[row_ix]:self.adjacency_matrix.indptr[row_ix+1] ] 
        #values = self.transition_matrix.data[self.transition_matrix.indptr[row_ix]:self.transition_matrix.indptr[row_ix+1]] 
        return [self.index_to_node[col_ix] for col_ix in column_indices]#, np.copy(values)

    def get_successors(self, node_id): #  -> Dict[str, Set[Node]]
        """Retuns a dict with sets of adjacent nodes."""
        assert node_id in self.nodes, "Unknown node"
        if node_id not in self.__successors.keys():
            self.__successors[node_id] = self.__get_successors(node_id)
        return self.__successors[node_id]
    def __str__(self):
        return "Network"
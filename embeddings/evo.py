from .abstract_embedding import AbstractEmbedding
import numpy as np
from collections import defaultdict

class EVO(AbstractEmbedding):
    def __init__(self, sequence_embedding_object):
        self.sequence_embedding_object = sequence_embedding_object
        self.sequence_embedding = sequence_embedding_object.embedding
        self.d = sequence_embedding_object.d
        self.nodes_embeddings = None
        self.f = None
        self.embedding = None
        self.nodes = None
        self.node_to_index = None
        self.index_to_node = None
        self.predicted_weighted_adjacency_matrix = None
        self.similarity_matrix = None

    def compute_embedding(self, f = np.mean):
        self.embedding = self.EVO_projection(f)

    def record_node_embeddings(self):
        """
            Creates a dictionary where each node is realted all the representations of nodes containing it in the HON
        """
        self.nodes_embeddings = defaultdict(list) 
        for seq in  self.sequence_embedding.keys():
            for node in seq:
                self.nodes_embeddings[node].append(np.array(self.sequence_embedding[seq])) 

    def EVO_projection(self,f): # sum only for now
        assert f in [np.max, np.mean, np.sum], "Better keep yourself to 'mean' and 'max'"
        
        if f is np.max:
            self.projection_function = "max"
        elif f is np.mean:
            self.projection_function = "mean"
        elif f is np.sum:
            self.projection_function = "sum"
        else: 
            assert False, "Only 'max', 'mean' and 'sum' for EVO projections. Computations should not have gotten here"

        if self.nodes_embeddings is None:
            self.record_node_embeddings()

        EVO_embedding = {}
        self.f = f #save function used to obtain embedding
        self.nodes = set()
        for node in self.nodes_embeddings.keys():
            self.nodes.add((node,))
            # n = len(self.nodes_embeddings[node])
            EVO_embedding[node] = f(self.nodes_embeddings[node], axis = 0) #/ n
        # probalby better to do the followign assignments outside the function
        self.index_to_node = {enum:node for enum,node in enumerate(self.nodes)}
        self.node_to_index = {node:enum for enum,node in self.index_to_node.items()}
        return {(k,):v for k,v in EVO_embedding.items()}


    def plot_embedding(self,edges, colors = None, plot_names = False): #node_to_index
        # assert nodes in input network are ok
        self.edges = edges
        # self.node_to_index = node_to_index
        super(EVO, self).plot_embedding(colors = colors, plot_names = plot_names)


    def __str__(self):
        return "EVO"

    def attributes_dictionary(self):
        return {
            "method":str(self),
            "d":self.d, 
            "projection_function":self.projection_function, 
            # "directed": self.sequence_embedding.network.directed,
            "sequence_embedding":self.sequence_embedding_object.attributes_dictionary(),
            }
         





from .random_walk_embedding import RandomWalkEmbedding
# from vincEmbedding.path_extraction.random_walk import Node2VecDynamics, WalkCollection
from networkx import Graph, DiGraph
from node2vec import Node2Vec
import numpy as np


class Node_to_Vec(RandomWalkEmbedding): 
    def __init__(self, d, network ,p, q,):
        super().__init__(d,network)
        self.p = p
        self.q = q
        self.network_nx = None
        self.directed = self.network.directed
        self.weighted = self.network.weighted

    def _simulate_walks(self):
        return super()._simulate_walks()


    def compute_embedding(self,num_walks = 80, walk_length = 40, window = 10): 
        """ 
        Computing the embedding through the node2vec package in github (eliorc)
        """   
        # assert False, "Incomplete function"
        self.num_walks = num_walks
        self.walk_length = walk_length 
        self.window = window        
        
        if self.directed:
            self.network_nx = DiGraph()
        else:
            self.network_nx = Graph()
        self.network_nx.add_nodes_from([str(self.node_to_index[node]) for node in self.network.nodes])
        if self.weighted: 
            self.network_nx.add_edges_from([
                (str(self.node_to_index[e[0]]), # str
                str(self.node_to_index[e[1]]), 
                {"weight":
                    self.network.adjacency_matrix[
                    self.node_to_index[e[0]],
                    self.node_to_index[e[1]]]}) for e in self.network.edges])
            node2vec = Node2Vec(
                graph = self.network_nx,
                dimensions=self.d,
                walk_length=walk_length,
                num_walks=num_walks,
                p = self.p,
                q = self.q,
                workers=1,
                weight_key = "weight"
                )
        else: 
            self.network_nx.add_edges_from([
                (str(self.node_to_index[e[0]]), # str
                str(self.node_to_index[e[1]])) for e in self.network.edges])
            node2vec = Node2Vec(
                graph = self.network_nx,
                dimensions=self.d,
                walk_length=walk_length,
                num_walks=num_walks,
                p = self.p,
                q = self.q,
                workers=1,
                )
            
        
        self.model = node2vec.fit(window=window, min_count=1, batch_words=4)

        ixs_in_vocabulary = set(self.model.wv.vocab.keys())
        self.embedding = {}
        # for k in self.network.nodes:
        #     if str(k) in nodes_in_vocabulary:
        for node_ix in ixs_in_vocabulary:
            #self.embedding[self.__from_str_to_tuple(node)] = self.model.wv.get_vector(node)
            self.embedding[
                self.network.index_to_node[int(node_ix)]
                ] = self.model.wv.get_vector(node_ix)
        nodes_in_vocabulary = [self.network.index_to_node[int(node_ix)] for node_ix in ixs_in_vocabulary]
        #add coordinates for nodes that were not observed through rw
        for node in set(self.network.nodes).difference(set(nodes_in_vocabulary)): # {self.__from_str_to_tuple(node) for node in nodes_in_vocabulary}
            print("Node \"{}\" is in no path and has been given random coordinates ".format(node))
            self.embedding[node] = np.random.rand(self.d)



    def __str__(self):
        return "Node_to_Vec"
    def attributes_dictionary(self):
        return {
            "method":str(self),
            "d":self.d, 
            "network_type":str(self.network),
            "directed":self.network.directed,
            "order":self.network.order,
            "p":self.p,
            "q":self.q,
            "n_walks": self.num_walks,
            "len_walks": self.walk_length,
            "window":self.window
            }

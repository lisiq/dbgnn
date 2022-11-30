from ast import Assert
import scipy as sp
import numpy as np
import os

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


from abc import ABC, abstractmethod




class AbstractEmbedding(ABC):
    def __init__(self,d, network):
        super().__init__()
        self.d = d
        # self.network = network
        self.network = network
        # self.transition_matrix = self.network.transition_matrix
        self.adjacency_matrix = self.network.adjacency_matrix
        
        # self.start_probability = # transition_matrix + start and end probabilities should give the same as whole network
        # self.end_probability = 

        # account for str node uids
        self.node_to_index = network.node_to_index#{((v,) if isinstance(v, str) else v):k  for v,k in network.nodes.index.items() } #self.network.nodes.index 
        self.index_to_node = {v:k for k,v in self.node_to_index.items()}
        self.nodes = set(self.index_to_node.values())

        us, vs = self.adjacency_matrix.nonzero()
        n_edges = len(us)
        self.edges = [(self.index_to_node[us[i]], self.index_to_node[vs[i]]) for i in range(n_edges)]  # TODO: this becomes network.edges

        self.embedding = None
        self.predicted_weighted_adjacency_matrix = None
        self.similarity_matrix = None
        


    
    @abstractmethod
    def compute_embedding():
        pass
    

    
    def get_vector(self, node_uid):
        assert node_uid in self.nodes, "Unknown uid"  # self.network.nodes.uids
        return self.embedding[node_uid]
    
    
    # choosing which similarity measure to use https://developers.google.com/machine-learning/clustering/similarity/measuring-similarity  
    # As shown in the link above, the measures are proportional to one another (Euclidea distance just goes in diff direction)
    def predict_weight(self, id_node_a, id_node_b):
        """
            Predicts the weight between two nodes.
        """
        if self.embedding is None:
            self.computeEmbedding()
        return np.dot(self.embedding[id_node_a], self.embedding[id_node_b])

    
    
    
    def get_predicted_weighted_adjacency_matrix(self):
        """
            computes predicted adjacency matrix given the embedding, bese on the function defined in predict_weight
        """
        if self.embedding is None:
            self.computeEmbedding()
        if self.predicted_weighted_adjacency_matrix is None:
            sorted_nodes = sorted(self.node_to_index.items(), key = lambda x:x[1])
            sorted_nodes = [i[0] for i in sorted_nodes] 
            self.predicted_weighted_adjacency_matrix = np.zeros((len(sorted_nodes),len(sorted_nodes)))
            for ix_row in range(len(sorted_nodes)):
                for ix_col in range(ix_row, len(sorted_nodes)):
                    weight = self.predict_weight(sorted_nodes[ix_row], sorted_nodes[ix_col])
                    self.predicted_weighted_adjacency_matrix[ix_row, ix_col] = weight
                    self.predicted_weighted_adjacency_matrix[ix_col, ix_row] = weight
        # remove meaningless self loops
        np.fill_diagonal(self.predicted_weighted_adjacency_matrix, 0)
        return self.predicted_weighted_adjacency_matrix


    def predict_adjacency_matrix(self,threshold):
        if self.predicted_weighted_adjacency_matrix is None:
            _ = self.get_predicted_weighted_adjacency_matrix()
            
        mask = self.predicted_weighted_adjacency_matrix >= threshold
        return mask*1

    def top_k_reconstructed(self,k): 
        """
        returns the top k edges with highest score
        """
        if self.predicted_weighted_adjacency_matrix is None:
            _ = self.get_predicted_weighted_adjacency_matrix()
        
        _, ix = np.unique(
            -self.predicted_weighted_adjacency_matrix,
            return_inverse = True) # returns the array values and their indexes (sorted by values)
        ix_us, ix_vs = (ix < k).reshape(self.predicted_weighted_adjacency_matrix.shape).nonzero()
        map_ix_uid = {v:k for k,v in self.node_to_index.items()}
        us = [map_ix_uid[ix_node] for ix_node in ix_us]
        vs = [map_ix_uid[ix_node] for ix_node in ix_vs]
        return [(u,v) for u,v in list(zip(us,vs)) ] #if u!=v
        
    def get_predicted_k_neighbors(self,node,k):
        """
            returns the k nodes closer to node with id "node" according to the embedding
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        node_ix = self.node_to_index[node]
        pairwise_distance_vector = self.similarity_matrix[node_ix]
        # remove node itself's entry
        pairwise_distance_vector = np.delete(
            pairwise_distance_vector,
            node_ix)
        pairwise_distance_vector = abs(pairwise_distance_vector)
        sorted_neighbors = sorted(enumerate(pairwise_distance_vector), key = lambda el : el[1], reverse = True)
        return [self.index_to_node[el[0]] for el in sorted_neighbors[:k]]

    def reconstruct_network(self, threshold):
        """
        Returns list of edges with similarity over threshold
        """
        if self.predicted_weighted_adjacency_matrix is None:
            _ = self.get_predicted_weighted_adjacency_matrix()
            
        mask = self.predicted_weighted_adjacency_matrix >= threshold
        row_indexes, col_indexes = np.where(np.triu(mask))
        map_ix_uid = {v:k for k,v in self.node_to_index.items()}
        us = [map_ix_uid[ix_node] for ix_node in row_indexes]
        vs = [map_ix_uid[ix_node] for ix_node in col_indexes]
        return [(u,v) for u,v in list(zip(us,vs)) if u!=v] + [(v,u) for u,v in list(zip(us,vs)) if u!=v]

    def predict_links(self, threshold):
        """
            Returns edgelist predicted given a threshold
            NB: only non observed edges
        """
        e_over_threshold = self.reconstruct_network(threshold)
        return [tup for tup in e_over_threshold if (tup not in self.edges and tup[::-1] not in self.edges)]
        
        
        
    
    
    def dimensionality_reduction(self ,d, method = 'PCA'):
        """
        Dimensionality Reduction of embedding object through PCA
        """
        assert self.d >= d, "requested dimensionality must be lower or equal to input"
        
        ordered_uids = [el[0] for el in sorted(self.node_to_index.items(), key = lambda x : x[1])]
        X = np.array([self.embedding[uid] for uid in ordered_uids])
        if method == 'PCA':
            pca = PCA(n_components=d, svd_solver='full')
            new_coordinates = pca.fit_transform(X)
        elif method == "TSNE":
            new_coordinates = TSNE(n_components=d).fit_transform(X)
        else:
            assert 2+2==5, "Method {} is not implemented".format(method)
            
        return new_coordinates


    def plot_embedding(self, figsize = [8,8], edgewidth=.1, pointsize = 100, method = "TSNE", colors = None, plot_names = False):
        if self.d == 1:
            assert False,  "Implemented only for dimension >= 2"
        elif self.d == 2: 
            coordinates = np.array(list(self.embedding.values())) 
        else: 
            coordinates = self.dimensionality_reduction(2, method = method)
        
        p_x = coordinates[:,0]
        p_y = coordinates[:,1]
        ix = self.node_to_index

        try:
            edges = [  [coordinates[ix[e[0]]], 
                        coordinates[ix[e[1]]] ]
                    for e in self.edges  
                    ]
        except KeyError:
            edges = [  [coordinates[ix[(e[0],)]],  
                        coordinates[ix[(e[1],)]] ]
                    for e in self.edges 
                    ]

        
        #adding edges
        lc = LineCollection(edges, linewidths=edgewidth)
        fig = plt.figure(figsize=figsize) 
        plt.gca().add_collection(lc)

        plt.axis('off')
        plt.xlim(min(p_x)-.1, max(p_x)+.1)
        plt.ylim(min(p_y)-.1, max(p_y)+.1)

        if colors is not None:
            cl = [colors[node] for node in self.node_to_index.keys()] # This has to take the order given by node_to_index because has have the same order as the node coordinates after dimensionality reduction
        else:
            cl = "k"

        plt.scatter(p_x, p_y, s= pointsize, zorder = 2,c = cl)
        


    # TODOs: change nome to "compute_similarity_matrix"
    def compute_similarity_matrix(self):
        if self.embedding is None:
            self.compute_embedding()

        self.similarity_matrix = np.zeros((len(self.embedding), len(self.embedding)))
        # node2ix = self.network.nodes.index
        for node1 in self.embedding:
            for node2 in self.embedding:
                self.similarity_matrix[
                    self.node_to_index[node1],
                    self.node_to_index[node2]] = self.predict_weight(node1, node2) #np.linalg.norm(self.embedding[node2] - self.embedding[node1])


    def plot_distance_matrix(self):
        """
            Plot of the pairwise distance between the nodes of the embedding
        """
        if self.embedding is None:
            self.compute_embedding()

        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        plt.matshow(self.similarity_matrix)


 
    def get_embedding_matrix(self):
        if self.embedding is None:
            self.compute_embedding()
        embedding_matrix = np.zeros((len(self.nodes),self.d))
        for node in self.nodes: 
            embedding_matrix[self.node_to_index[node],:] = self.embedding[node]
        return embedding_matrix
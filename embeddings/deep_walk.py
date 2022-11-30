
from .node2vec import Node_to_Vec


# JUST FOR UNIFORMING CODE AND TREATEMENT OF DIRECTED NETWORKS
class DeepWalk(Node_to_Vec):
    def __init__(self, d, network):
        super().__init__(d,network, p =1,q=1)
        
    
    def __str__(self):
        return "DeepWalk"
    def attributes_dictionary(self):
        return {
            "method":str(self),
            "d":self.d, 
            "network_type":str(self.network),
            "directed":self.network.directed,
            "order":self.network.order,
            "n_walks": self.num_walks,
            "len_walks": self.walk_length,
            "window":self.window
            }


# from dynamics import FirstOrderDynamics
# from sklearn.preprocessing import normalize
# class DeepWalk(RandomWalkEmbedding):
#     def __init__(self, d, network):
#         super().__init__(d,network)
        
#     def _simulate_walks(self):
#         walk_dynamics = FirstOrderDynamics(
#             normalize(self.adjacency_matrix, norm='l1', axis=1),
#             node_to_index=self.node_to_index,
#             index_to_node=self.index_to_node
#             )
#         wc = WalkCollection(
#             walk_dynamics = walk_dynamics,
#             n_walks = self.num_walks,
#             len_walks = self.walk_length
#             )
#         self.walks = wc.walks
    
#     def __str__(self):
#         return "DeepWalk"
#     def attributes_dictionary(self):
#         return {
#             "method":str(self),
#             "d":self.d, 
#             "network_type":str(self.network),
#             "order":self.network.order,
#             "n_walks": self.num_walks,
#             "len_walks": self.walk_length,
#             "window":self.window,
#             "directed":self.network.directed
#             }

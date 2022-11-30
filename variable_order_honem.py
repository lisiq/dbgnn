from network import Network_light

class VariableOrderNetwork_Xu_light(Network_light):
    def __init__(self,adjacency_matrix,node_to_index, directed = True):
        super().__init__(
            adjacency_matrix,
            node_to_index)
        self.order = max([len(k) for k in node_to_index.keys()])
        self.directed = directed #not check_symmetric(self.adjacency_matrix) # HEAVY
    def __str__(self):
        return "VariableOrder_XuChawla"
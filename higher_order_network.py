from network import Network_light
class HigherOrderNetwork_light(Network_light):
    def __init__(
        self,
        ho_adjacency_matrix,
        node_to_index_ho,
        network,
        directed = True
        ):
        """
        Higher-order network. Corresponing first order netwrok also in input, and goes through consistency checks.  
        """
        # assert nodes are tuples
        super().__init__(
            ho_adjacency_matrix,
            node_to_index_ho) # False
        # assert type of input network is correct
        #
        assert len(set([len(k) for k in node_to_index_ho.keys()])) == 1, "It seems like node_to_index_ho contains nodes from multiple orders"
        self.order = list(set([len(k) for k in node_to_index_ho.keys()]))[0]
        self.network = network
        self.ho_node_to_fo_index = self.__create_mapping_to_first_order()
        #
        self.directed = directed 
        self.null_model = False


    def __create_mapping_to_first_order(self):
        ho_node_to_fo_index = dict()
        # mapping between hon and first order
        for ho_node in self.node_to_index.keys():
            ho_node_to_fo_index[ho_node] = self.network.node_to_index[(ho_node[-1],)]
        return ho_node_to_fo_index
    def __str__(self):
        return "HigherOrderNetwork"

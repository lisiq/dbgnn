import torch 
import torch_geometric
from torch_geometric.nn import MessagePassing


# class BipartiteGraphOperator(MessagePassing):
#     def __init__(self, in_ch, out_ch):
#         super(BipartiteGraphOperator, self).__init__('add')
#         self.lin = torch.nn.Linear(in_ch*2, out_ch)

#     def forward(self, x, assign_index, N, M):
#         return self.propagate(assign_index, size=(N, M), x=x)

#     def message(self, x_i, x_j):
#         return self.lin(torch.cat([x_i, x_j], dim=1))

class BipartiteGraphOperator(torch_geometric.nn.MessagePassing):
    def __init__(self, in_ch, out_ch):
        super(BipartiteGraphOperator, self).__init__('add')
        self.lin1 = torch.nn.Linear(in_ch, out_ch)
        self.lin2 = torch.nn.Linear(in_ch, out_ch)

    def forward(self, x, bipartite_index, N, M):
        x = (self.lin1(x[0]), self.lin2(x[1]))
        return self.propagate(bipartite_index, size=(N, M), x=x)


import torch
import torch_geometric
# from .bipartite_operator import BipartiteGraphOperator

class HO_GCN(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        num_features = [60, 32], # ho ,fo
        hidden_dims = [16,32,32], #[16, 32, 32,64,64],
        p_dropout = 0.4
        ):
        super().__init__()
        #self.device = "cpu" # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        current_dim = num_features
        self.p_dropout = p_dropout
        
        # order 2
        self.layer_2_2_0 = torch_geometric.nn.GCNConv(self.num_features[0], self.hidden_dims[0])
        self.layer_2_2_1 = torch_geometric.nn.GCNConv(self.hidden_dims[0], self.hidden_dims[1])

        # order 1
        self.layer_1_1_0 = torch_geometric.nn.GCNConv(self.num_features[1], self.hidden_dims[0])
        self.layer_1_1_1 = torch_geometric.nn.GCNConv(self.hidden_dims[0], self.hidden_dims[1])

        self.layer_2_1 = BipartiteGraphOperator(self.hidden_dims[1], self.hidden_dims[2])
        
        # mlp
        self.mlp = torch.nn.Linear(self.hidden_dims[2], num_classes)
        
        
    
    def forward(self, data, device): # index_list, num_ho_fo_nodes):#, x_n2v):
        import torch.nn.functional as F
        # higher_order_adjacency_matrix = higher_order_network.adjacency_matrix
        ho_index_list = data.edge_index #torch.tensor(tuple(zip(*higher_order_adjacency_matrix.nonzero()))).long().T
        ho_weights = data.edge_weight #torch.tensor(higher_order_adjacency_matrix.data).float()
        # edge index for biparties layer
        ho_index_to_fo_index = data.edge_index_hon_to_fon
        
        # 2-2
        x = data.x_ho
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = torch.nn.functional.elu(self.layer_2_2_0(x, ho_index_list, ho_weights))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = torch.nn.functional.elu(self.layer_2_2_1(x, ho_index_list, ho_weights))
        x = F.dropout(x, p=self.p_dropout, training=self.training)

        fo_index_list = data.edge_index_fo
        fo_weights = data.edge_weight_fo

        # 1-1
        x1 = data.x_fo
        x1 = F.dropout(x1, p=self.p_dropout, training=self.training)
        x1 = torch.nn.functional.elu(self.layer_1_1_0(x1, fo_index_list, fo_weights))
        x1 = F.dropout(x1, p=self.p_dropout, training=self.training)
        x1 = torch.nn.functional.elu(self.layer_1_1_1(x1, fo_index_list, fo_weights))
        x1 = F.dropout(x1, p=self.p_dropout, training=self.training)

        # 2-1
        num_fo_nodes = data.num_nodes #higher_order_network.network.n_nodes #num_ho_fo_nodes[1]
        num_ho_nodes = data.num_ho_nodes #higher_order_network.n_nodes #num_ho_fo_nodes[0]
        # x_right = torch.empty(num_fo_nodes,x.shape[1]).to(device)
        # torch.nn.init.xavier_uniform_(x_right)
        # x_right = torch.ones(num_fo_nodes,x.shape[1]).to(device)
        x = torch.nn.functional.elu(self.layer_2_1((x, x1), ho_index_to_fo_index, N = num_ho_nodes, M= num_fo_nodes))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        # # 1-1
        # print('bla')
        x = self.mlp(x)
           
        # out = torch.nn.functional.softmax(x, dim=1)
            
        return x
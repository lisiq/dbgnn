import os 
#
import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dims =[16, 32,32],
        p_dropout = 0.4
        ):
        super(GCN, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        current_dim = num_features
        self.p_dropout = p_dropout
        
        # order 2
        self.layer_2_2_0 = torch_geometric.nn.GCNConv(self.num_features, self.hidden_dims[0])
        self.layer_2_2_1 = torch_geometric.nn.GCNConv(self.hidden_dims[0], self.hidden_dims[1])
        self.layer_2_2_2 = torch_geometric.nn.GCNConv(self.hidden_dims[1], self.hidden_dims[2])
                
        # mlp
        self.mlp = torch.nn.Linear(self.hidden_dims[2], num_classes)
                
    
    def forward(self, data):
        import torch.nn.functional as F
        # adjacency_matrix = network.adjacency_matrix
        index_list = data.edge_index  #torch.tensor(tuple(zip(*adjacency_matrix.nonzero()))).long().T
        weights = data.edge_weight #torch.tensor(adjacency_matrix.data).float()
        
        # 2-2
        x = data.x
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = torch.nn.functional.elu(self.layer_2_2_0(x, index_list, weights))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = torch.nn.functional.elu(self.layer_2_2_1(x, index_list, weights))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = torch.nn.functional.elu(self.layer_2_2_2(x, index_list, weights))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        
        x = self.mlp(x)
           
        # out = torch.nn.functional.softmax(x, dim=1)

        return x
    
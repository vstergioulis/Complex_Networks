import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, NNConv, GATv2Conv
from torch.nn import Linear, Sequential, ReLU

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())


class GCN(torch.nn.Module):
    def __init__(self, input_, hidden_channels, output_):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GCNConv(input_, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, output_)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.6, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        
        
        x = self.out(x)
        return x

class GAT(torch.nn.Module):
    def __init__(self, input_, hidden_channels, output_, heads=4):
        super(GAT, self).__init__()
        torch.manual_seed(42)

        # GAT layers with attention heads
        self.conv1 = GATConv(input_, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)

        # Final output layer (same as GCN)
        self.out = Linear(hidden_channels * heads, output_)

    def forward(self, x, edge_index):
        # First GAT layer
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        # Second GAT layer
        x = self.conv2(x, edge_index)

        # Output layer
        x = self.out(x)
        return x #F.softmax(x, dim=1)

class GCN_V2(torch.nn.Module):
    def __init__(self, input_ , hidden_channels, edge_dim, output_):
        super(GCN_V2, self).__init__()
        torch.manual_seed(42)

        # Edge-conditioned MLPs
        nn1 = Sequential(Linear(edge_dim, 128), ReLU(), Linear(128, input_ * hidden_channels))
        nn2 = Sequential(Linear(edge_dim, 128), ReLU(), Linear(128, hidden_channels * hidden_channels))

        # NNConv layers
        self.conv1 = NNConv(input_, hidden_channels, nn1, aggr='mean')
        self.conv2 = NNConv(hidden_channels, hidden_channels, nn2, aggr='mean')

        self.out = Linear(hidden_channels, output_)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        #x = F.relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)

        x = self.out(x)
        return x #F.log_softmax(x, dim=1)
    
    
class GAT_V2(torch.nn.Module):
    def __init__(self, input_, hidden_channels, edge_dim, output_):
        super(GAT_V2, self).__init__()
        self.conv1 = GATv2Conv(input_, hidden_channels, heads=4, concat=True, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=4,  concat=True , edge_dim=edge_dim)
        self.lin = Linear(hidden_channels * 4, output_)

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)


        x = self.conv2(x, edge_index, edge_attr)
        
        return self.lin(x) #F.log_softmax(self.lin(x), dim=1)

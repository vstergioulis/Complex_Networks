from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch

def load_dataset(name = 'Cora'):
    
    
    return Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures()) 



def edge_attributes(edge_attr = 'all', edge_index = None, x = None):
    
    row, col = edge_index
    abs_ = torch.abs(x[row] - x[col])
    concat_ = torch.cat([x[row], x[col]], dim=1)
    hadamard = x[row] * x[col]
    
    cos_sim = torch.nn.functional.cosine_similarity(x[row], x[col], dim=1).unsqueeze(1)
    cos_ = cos_sim.repeat(1, x.size(1))
    
    if edge_attr == 'abs':
        f = abs_
        
    elif edge_attr == 'concat':
        f = concat_
        
    elif edge_attr == 'had':
        f = hadamard
        
    elif edge_attr == 'cos':
        f = cos_
        
    else:
        f = torch.cat([abs_, hadamard ,cos_], dim=1)
        
    return f
        
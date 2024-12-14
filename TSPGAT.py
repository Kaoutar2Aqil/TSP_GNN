import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

class TSPGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TSPGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)  # 4 têtes d'attention
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))  # Convolution avec attention 1
        x = F.elu(self.conv2(x, edge_index))  # Convolution avec attention 2
        x = self.conv3(x, edge_index)  # Sortie finale
        return x
from torch_geometric.data import Data
import numpy as np

# Exemple de 5 villes avec coordonnées (x, y)
node_features = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float)
edges = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 0, 3, 2, 1, 4]], dtype=torch.long)  # Arêtes

# Graphe PyTorch Geometric
data = Data(x=node_features, edge_index=edges)

# Modèle GNN
model = TSPGAT(in_channels=2, hidden_channels=16, out_channels=2)
out = model(data)
print("Sortie des nœuds :", out)

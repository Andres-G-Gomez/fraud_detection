from src.models.graph_detector import load_processed_data
import torch

data = load_processed_data('data/processed/features_with_labels', 'data/processed/edges', torch.device('cpu'))
num_nodes = data.num_nodes
print('num_nodes', num_nodes)
print('x shape', data.x.shape)
print('y distribution', torch.bincount(data.y).tolist())
indices = torch.randperm(num_nodes)
train_size = int(0.7 * num_nodes)
val_size = int(0.15 * num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[indices[:train_size]] = True
val_mask[indices[train_size:train_size + val_size]] = True
test_mask[indices[train_size + val_size:]] = True
print('train', train_mask.sum().item(), 'val', val_mask.sum().item(), 'test', test_mask.sum().item(), 'total', train_mask.sum().item() + val_mask.sum().item() + test_mask.sum().item())

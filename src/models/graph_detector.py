"""
Graph Neural Network Model for Bitcoin Transaction Fraud Detection

This module implements:
1. A Graph Convolutional Network (GCN) using PyTorch Geometric
2. Focal Loss to handle severe class imbalance (~2% illicit)
3. MLflow tracking for hyperparameters and metrics
4. Model training and evaluation pipeline
"""

import argparse
import logging 
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score
import mlflow
import mlflow.pytorch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    
    Attributes:
        alpha: Weight for the positive / rare class
        gamma: Focusing parameter (higher gamma = focus on hard examples)
    """
    
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weight for positive class (fraud). Set this to the fraction of
                legitimate samples in the dataset to up-weight rare fraud cases.
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            predictions: Model logits [batch_size] or [batch_size, 1]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Focal loss value
        """
        # Compute BCE loss
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets.float(), reduction='none')
        
        # Compute probability of the true class
        p_t = torch.exp(-ce_loss)
        
        # Apply standard class-balanced focal weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss
        
        return focal_loss.mean()


class GCNFraudDetector(nn.Module):
    """
    Graph Convolutional Network for fraud detection in Bitcoin transactions.
    
    Attributes:
        in_channels: Number of input node features
        hidden_channels: Number of hidden layer channels
        out_channels: Number of output channels (1 for binary classification)
        num_layers: Number of GCN layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        """
        Initialize GCN model.
        
        Args:
            in_channels: Input feature dimension (166 for Elliptic)
            hidden_channels: Hidden layer dimensions
            out_channels: Output dimensions (1 for binary fraud classification)
            num_layers: Number of GCN layers
            dropout: Dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GCN.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Output predictions [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation)
        x = self.convs[-1](x, edge_index)
        
        return x.squeeze(-1)  # [num_nodes]


def load_processed_data(
    features_path: str,
    edges_path: str,
    device: torch.device
) -> Data:
    """
    Load processed Parquet files and construct PyTorch Geometric Data object.
    
    Args:
        features_path: Path to features Parquet directory
        edges_path: Path to edges Parquet directory
        device: Torch device (cpu or cuda)
        
    Returns:
        PyTorch Geometric Data object
    """
    logger.info("Loading processed data...")
    
    import pandas as pd
    
    # Load features (from parquet partition)
    features_files = list(Path(features_path).glob("part-*.parquet"))
    features_list = [pd.read_parquet(f) for f in sorted(features_files)]
    features_df = pd.concat(features_list, ignore_index=True)
    
    # Load edges
    edges_files = list(Path(edges_path).glob("part-*.parquet"))
    edges_list = [pd.read_parquet(f) for f in sorted(edges_files)]
    edges_df = pd.concat(edges_list, ignore_index=True)
    
    logger.info(f"Loaded {len(features_df)} nodes and {len(edges_df)} edges")
    
    # Extract features and labels
    tx_ids = features_df['tx_id'].values
    labels = features_df['label'].astype(np.int64).values
    
    # Feature columns (skip tx_id and label)
    feature_cols = [col for col in features_df.columns if col not in ['tx_id', 'label']]
    features = features_df[feature_cols].values.astype(np.float32)
    
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Labels distribution: {np.bincount(labels)}")
    
    # Create ID to index mapping
    id_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
    
    # Convert edge IDs to indices
    edge_source = edges_df['source_id'].map(id_to_idx).values
    edge_target = edges_df['target_id'].map(id_to_idx).values
    
    # Filter out edges with missing nodes (shouldn't happen, but safe)
    valid_mask = (edge_source >= 0) & (edge_target >= 0) & \
                 (edge_source < len(tx_ids)) & (edge_target < len(tx_ids))
    edge_source = edge_source[valid_mask]
    edge_target = edge_target[valid_mask]
    
    edge_index = torch.tensor(np.vstack([edge_source, edge_target]), dtype=torch.long)
    
    # Create PyG Data object
    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
        num_nodes=len(tx_ids)
    )
    
    data = data.to(device)
    
    logger.info(f"Data object created: {data}")
    
    return data


def train_epoch(
    model: GCNFraudDetector,
    data: Data,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_mask: torch.Tensor
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: GCN model
        data: PyG Data object
        criterion: Loss function
        optimizer: Optimizer
        device: Torch device
        train_mask: Boolean mask for training nodes
        
    Returns:
        Average training loss
    """
    model.train()
    optimizer.zero_grad()
    
    logits = model(data.x, data.edge_index)
    loss = criterion(logits[train_mask], data.y[train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(
    model: GCNFraudDetector,
    data: Data,
    mask: torch.Tensor,
    device: torch.device,
    threshold=0.6331
) -> Tuple[float, float, float, float]:
    """
    Evaluate model on a subset of nodes.
    
    Args:
        model: GCN model
        data: PyG Data object
        mask: Boolean mask for evaluation nodes
        device: Torch device
        
    Returns:
        Tuple of (Precision, Recall, PR-AUC, ROC-AUC)
    """
    model.eval()
    
    logits = model(data.x, data.edge_index)
    predictions = torch.sigmoid(logits[mask])
    targets = data.y[mask].cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    
    # Compute metrics
    pred_labels = (predictions_np >= threshold).astype(int)
    
    precision = precision_score(targets, pred_labels, zero_division=0)
    recall = recall_score(targets, pred_labels, zero_division=0)
    pr_auc = average_precision_score(targets, predictions_np)
    roc_auc = roc_auc_score(targets, predictions_np)
    
    return precision, recall, pr_auc, roc_auc


def train_gnn(
    data_dir: str = "data/processed",
    model_output_dir: str = "models",
    epochs: int = 50,
    learning_rate: float = 0.001,
    hidden_channels: int = 64,
    dropout: float = 0.5,
    use_cuda: bool = False,
    log_every: int = 1,
    patience: int = 7,
    min_delta: float = 1e-4
) -> None:
    """
    Main training pipeline with MLflow tracking.
    
    Args:
        data_dir: Path to processed data directory
        model_output_dir: Directory to save model weights
        epochs: Number of training epochs
        learning_rate: Learning rate
        hidden_channels: Hidden channel dimension
        dropout: Dropout rate
        use_cuda: Whether to use CUDA
        log_every: Log metrics every N epochs
    """
    # Set up device
    device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    features_path = f"{data_dir}/features_with_labels"
    edges_path = f"{data_dir}/edges"
    data = load_processed_data(features_path, edges_path, device)
    
    # Initialize MLflow run
    mlflow.set_experiment("AML_GCN_Training")
    
    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params({
            "epochs": epochs,
            "learning_rate": learning_rate,
            "hidden_channels": hidden_channels,
            "dropout": dropout,
            "log_every": log_every,
            "model_type": "GCN",
            "loss_function": "FocalLoss",
            "device": str(device)
        })
        
        logger.info("=" * 60)
        logger.info("Starting GNN Training with MLflow Tracking")
        logger.info("=" * 60)
        
        # Create model
        model = GCNFraudDetector(
            in_channels=data.x.shape[1],
            hidden_channels=hidden_channels,
            out_channels=1,
            num_layers=3,
            dropout=dropout
        ).to(device)
        
        logger.info(f"Model: {model}")
        
        # Create splits (use 70% train, 15% val, 15% test)
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        train_size = int(0.7 * num_nodes)
        val_size = int(0.15 * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        logger.info(f"Train/Val/Test split: {train_size}/{val_size}/{num_nodes - train_size - val_size}")
        
        # Compute class weighting for focal loss: weight the rare fraud class by the
        # proportion of legitimate samples in the dataset.
        label_counts = torch.bincount(data.y)
        num_safe = int(label_counts[0].item())
        num_fraud = int(label_counts[1].item())
        total = num_safe + num_fraud
        alpha = float(num_safe) / float(total) if total > 0 else 0.5
        logger.info(f"Using focal loss alpha={alpha:.4f} ({num_safe}/{total} legitimate/fraud)")

        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        best_pr_auc = -1.0
        best_epoch = 0
        epochs_without_improvement = 0
        
        # Training loop
        for epoch in range(epochs):
            train_loss = train_epoch(model, data, criterion, optimizer, device, train_mask)
            
            # Evaluate
            val_prec, val_recall, val_pr_auc, val_roc_auc = evaluate(
                model, data, val_mask, device
            )
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_precision": val_prec,
                "val_recall": val_recall,
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": val_roc_auc
            }, step=epoch)
            
            if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
                message = (
                    f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                    f"Val Prec: {val_prec:.4f} | Val Rec: {val_recall:.4f} | "
                    f"PR-AUC: {val_pr_auc:.4f} | ROC-AUC: {val_roc_auc:.4f}"
                )
                logger.info(message)
                print(message, file=sys.stdout, flush=True)
            
            # Save best model and reset early stopping counter when PR-AUC improves
            if val_pr_auc > best_pr_auc + min_delta:
                best_pr_auc = val_pr_auc
                best_epoch = epoch
                epochs_without_improvement = 0
                model_path = f"{model_output_dir}/best_model.pt"
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved best model with PR-AUC: {best_pr_auc:.4f}")
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement in PR-AUC for {epochs_without_improvement}/{patience} epochs")

            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1} with best PR-AUC {best_pr_auc:.4f} at epoch {best_epoch+1}")
                break
        
        # Evaluate on test set
        test_prec, test_recall, test_pr_auc, test_roc_auc = evaluate(
            model, data, test_mask, device
        )
        
        logger.info("=" * 60)
        logger.info("Final Test Set Metrics:")
        logger.info(f"  Precision: {test_prec:.4f}")
        logger.info(f"  Recall: {test_recall:.4f}")
        logger.info(f"  PR-AUC: {test_pr_auc:.4f}")
        logger.info(f"  ROC-AUC: {test_roc_auc:.4f}")
        logger.info(f"  Best Epoch: {best_epoch}")
        logger.info("=" * 60)
        
        # Log final metrics
        mlflow.log_metrics({
            "test_precision": test_prec,
            "test_recall": test_recall,
            "test_pr_auc": test_pr_auc,
            "test_roc_auc": test_roc_auc,
            "best_epoch": best_epoch
        })
        
        # Save model with MLflow
        mlflow.pytorch.log_model(model, "model")
        
        logger.info(f"MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GCN model for Bitcoin transaction fraud detection"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory (default: data/processed)"
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default="models",
        help="Directory to save model weights (default: models)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer (default: 0.001)"
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Number of hidden channels in GCN layers (default: 64)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate (default: 0.5)"
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Enable CUDA GPU training (default: CPU only)"
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log metrics every N epochs (default: 1)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience on validation PR-AUC (default: 7)"
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum PR-AUC improvement to reset early stopping (default: 1e-4)"
    )
    
    args = parser.parse_args()
    
    train_gnn(
        data_dir=args.data_dir,
        model_output_dir=args.model_output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        use_cuda=args.use_cuda,
        log_every=args.log_every,
        patience=args.patience,
        min_delta=args.min_delta
    )

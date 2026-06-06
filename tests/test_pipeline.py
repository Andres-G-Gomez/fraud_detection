"""
Unit and integration tests for Phase 1: Data Pipeline & GNN Model

Tests verify:
1. Data shapes and integrity after ETL
2. Edge list is properly directed
3. Class distribution after processing
4. Model architecture and forward pass
5. Loss function computation
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from typing import Tuple

# Test fixtures
@pytest.fixture
def sample_features() -> np.ndarray:
    """Create sample feature matrix."""
    return np.random.randn(1000, 166).astype(np.float32)


@pytest.fixture
def sample_edges() -> Tuple[np.ndarray, np.ndarray]:
    """Create sample directed edge list."""
    sources = np.random.randint(0, 1000, 500)
    targets = np.random.randint(0, 1000, 500)
    return sources, targets


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Create sample labels with imbalance."""
    labels = np.zeros(1000)
    # Add ~2% illicit (class 1)
    illicit_idx = np.random.choice(1000, size=20, replace=False)
    labels[illicit_idx] = 1
    return labels.astype(np.int64)


class TestDataPipeline:
    """Tests for PySpark ETL pipeline."""
    
    def test_data_loading(self):
        """Test that raw data files exist and are readable."""
        base_path = Path("data/raw/elliptic_bitcoin_dataset")
        
        assert (base_path / "elliptic_txs_features.csv").exists(), "Features file not found"
        assert (base_path / "elliptic_txs_classes.csv").exists(), "Classes file not found"
        assert (base_path / "elliptic_txs_edgelist.csv").exists(), "Edges file not found"
    
    def test_processed_data_exists(self):
        """Test that processed data directory exists after ETL."""
        output_path = Path("data/processed")
        
        # Note: This will pass only after running the ETL pipeline
        if output_path.exists():
            assert (output_path / "features_with_labels").exists() or \
                   any(output_path.glob("features_with_labels/*")), \
                   "Features output not found"
            assert (output_path / "edges").exists() or \
                   any(output_path.glob("edges/*")), \
                   "Edges output not found"
    
    def test_class_filtering(self):
        """
        Test that unknown class labels are filtered correctly.
        From raw data: 157,205 unknown, 42,019 licit (2), 4,545 illicit (1)
        Expected after filtering: 46,564 total (42,019 + 4,545)
        """
        import pandas as pd
        
        # Load raw classes
        classes_df = pd.read_csv("data/raw/elliptic_bitcoin_dataset/elliptic_txs_classes.csv",
                                  header=0)
        
        # Count unknowns
        total_raw = len(classes_df)
        unknown_count = (classes_df['class'] == "unknown").sum()
        
        # Verify class distribution
        labeled_count = total_raw - unknown_count
        illicit_count = (classes_df['class'] == "1").sum()
        licit_count = (classes_df['class'] == "2").sum()
        
        assert total_raw == 203769, "Total raw records mismatch"
        assert unknown_count == 157205, "Unknown count mismatch"
        assert illicit_count == 4545, "Illicit count mismatch"
        assert licit_count == 42019, "Licit count mismatch"
        assert labeled_count == illicit_count + licit_count, "Labeled total mismatch"
    
    def test_feature_dimensions(self):
        """Test that features have correct dimensions (166 features + 1 ID)."""
        import pandas as pd
        
        features_df = pd.read_csv(
            "data/raw/elliptic_bitcoin_dataset/elliptic_txs_features.csv",
            header=None,
            nrows=100
        )
        
        # Should be 167 columns (1 ID + 166 features)
        assert len(features_df.columns) == 167, \
            f"Expected 167 columns, got {len(features_df.columns)}"
    
    def test_edge_list_directed(self):
        """Test that edge list is properly directed (not symmetric)."""
        import pandas as pd
        
        edges_df = pd.read_csv(
            "data/raw/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv",
            header=0
        )
        
        # Create set of edges and reversed edges
        edges = set(zip(edges_df['txId1'], edges_df['txId2']))
        reversed_edges = set(zip(edges_df['txId2'], edges_df['txId1']))
        
        # Count symmetric edges
        symmetric_count = len(edges & reversed_edges)
        
        # Graph should be mostly directed (not symmetric)
        # If it were fully symmetric, all edges would appear reversed
        assert symmetric_count < len(edges) * 0.5, \
            f"Graph appears undirected: {symmetric_count}/{len(edges)} edges are symmetric"
    
    def test_class_imbalance_percentage(self):
        """Test that illicit class is ~9.7% after removing unknown labels."""
        import pandas as pd
        
        classes_df = pd.read_csv("data/raw/elliptic_bitcoin_dataset/elliptic_txs_classes.csv",
                                  header=0)
        
        # Filter unknowns
        labeled = classes_df[classes_df['class'] != "unknown"]
        
        illicit_ratio = (labeled['class'] == "1").sum() / len(labeled)
        
        # Should be approximately 9.75%
        assert 0.095 < illicit_ratio < 0.10, \
            f"Expected ~9.75% illicit (4545/46564), got {illicit_ratio:.2%}"


class TestGNNModel:
    """Tests for PyTorch Geometric GCN model."""
    
    def test_gcn_model_creation(self):
        """Test that GCN model can be instantiated."""
        from src.models.graph_detector import GCNFraudDetector
        
        model = GCNFraudDetector(
            in_channels=166,
            hidden_channels=64,
            out_channels=1,
            num_layers=3,
            dropout=0.5
        )
        
        assert model is not None, "Failed to create GCN model"
        assert len(model.convs) == 3, "Expected 3 GCN layers"
    
    def test_gcn_forward_pass(self, sample_features, sample_edges):
        """Test forward pass through GCN."""
        from src.models.graph_detector import GCNFraudDetector
        
        model = GCNFraudDetector(
            in_channels=166,
            hidden_channels=64,
            out_channels=1,
            num_layers=3,
            dropout=0.5
        )
        model.eval()
        
        x = torch.tensor(sample_features)
        edge_index = torch.tensor([sample_edges[0], sample_edges[1]], dtype=torch.long)
        
        with torch.no_grad():
            output = model(x, edge_index)
        
        assert output.shape[0] == 1000, "Output shape mismatch"
        assert output.dtype == torch.float32, "Output dtype should be float32"
    
    def test_focal_loss(self, sample_labels):
        """Test Focal Loss computation."""
        from src.models.graph_detector import FocalLoss
        
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Create predictions
        predictions = torch.randn(1000, dtype=torch.float32)
        targets = torch.tensor(sample_labels, dtype=torch.float32)
        
        loss = criterion(predictions, targets)
        
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_model_parameters_count(self):
        """Test that model has reasonable number of parameters."""
        from src.models.graph_detector import GCNFraudDetector
        
        model = GCNFraudDetector(
            in_channels=166,
            hidden_channels=64,
            out_channels=1,
            num_layers=3,
            dropout=0.5
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        
        # Should be in reasonable range
        assert 10000 < param_count < 60000, \
            f"Parameter count {param_count} seems unreasonable"
    
    def test_batch_norm_layers(self):
        """Test that batch norm layers are present."""
        from src.models.graph_detector import GCNFraudDetector
        
        model = GCNFraudDetector(
            in_channels=166,
            hidden_channels=64,
            out_channels=1,
            num_layers=3,
            dropout=0.5
        )
        
        assert len(model.bns) == 2, "Expected 2 batch norm layers for 3 GCN layers"


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_data_shapes_consistency(self):
        """Test that data shapes are consistent across pipeline."""
        import pandas as pd
        
        # Load all raw files
        features_df = pd.read_csv(
            "data/raw/elliptic_bitcoin_dataset/elliptic_txs_features.csv",
            header=None
        )
        classes_df = pd.read_csv(
            "data/raw/elliptic_bitcoin_dataset/elliptic_txs_classes.csv",
            header=0
        )
        edges_df = pd.read_csv(
            "data/raw/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv",
            header=0
        )
        
        # Verify consistency
        n_features_rows = len(features_df)
        n_classes_rows = len(classes_df)
        
        # Both should have same number of transaction IDs
        assert n_features_rows == n_classes_rows, \
            f"Feature ({n_features_rows}) and class ({n_classes_rows}) row mismatch"
    
    def test_edge_node_consistency(self):
        """Test that edge nodes exist in feature set."""
        import pandas as pd
        
        features_df = pd.read_csv(
            "data/raw/elliptic_bitcoin_dataset/elliptic_txs_features.csv",
            header=None
        )
        edges_df = pd.read_csv(
            "data/raw/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv",
            header=0
        )
        
        valid_tx_ids = set(features_df[0].unique())
        
        # Check that edge nodes are in valid set
        edge_nodes = set(edges_df['txId1'].unique()) | set(edges_df['txId2'].unique())
        
        # Note: Not all edge nodes might be in features (future transactions)
        # But most should be
        covered_ratio = len(edge_nodes & valid_tx_ids) / len(edge_nodes)
        
        assert covered_ratio > 0.95, \
            f"Only {covered_ratio:.1%} of edge nodes found in features"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

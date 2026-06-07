#!/usr/bin/env python
"""Plot Precision-Recall curve for the validation set using a trained GraphSAGE model.

Loads the processed Elliptic dataset and a saved GraphSAGE state dict, computes
validation predictions, and plots the Precision-Recall curve. The optimal
classification threshold is selected by maximising F1 score on the curve.

Usage:
    python plot_graphsage_pr_curve.py
    python plot_graphsage_pr_curve.py --model-path models/best_graphsage_model.pt \\
                                      --aggregator max --hidden-channels 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

repo_root = Path(__file__).resolve().parent
sys.path.append(str(repo_root))

from src.models.graphsage_detector import GraphSAGEFraudDetector, load_processed_data


def create_split_masks(
    num_nodes: int, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recreate the 70/15/15 train/val/test masks with a fixed seed."""
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_nodes, generator=rng)
    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size : train_size + val_size]] = True
    test_mask[indices[train_size + val_size :]] = True

    return train_mask, val_mask, test_mask


def plot_precision_recall(
    recall: np.ndarray,
    precision: np.ndarray,
    threshold: float,
    precision_at_threshold: float,
    recall_at_threshold: float,
    average_precision: float,
    f1_at_threshold: float,
    aggregator: str,
    image_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where="post", label="PR curve", color="#2ca02c")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="#2ca02c")
    plt.scatter(
        [recall_at_threshold],
        [precision_at_threshold],
        color="red",
        zorder=5,
        label=f"Best F1 threshold = {threshold:.4f}",
    )

    plt.title(f"GraphSAGE ({aggregator}) — Validation Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.text(
        0.55,
        0.15,
        f"Avg Precision (PR-AUC): {average_precision:.4f}\n"
        f"Optimal threshold:      {threshold:.4f}\n"
        f"Precision @ threshold:  {precision_at_threshold:.4f}\n"
        f"Recall    @ threshold:  {recall_at_threshold:.4f}\n"
        f"F1        @ threshold:  {f1_at_threshold:.4f}",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="#888888"),
    )

    image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot validation PR curve for a trained GraphSAGE model."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory (default: data/processed)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_graphsage_model.pt",
        help="Path to saved GraphSAGE state dict (default: models/best_graphsage_model.pt)",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="graphsage_pr_curve.png",
        help="Output path for the PR curve image (default: graphsage_pr_curve.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val/test split (default: 42)",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Hidden channel size matching the trained model (default: 64)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability matching the trained model (default: 0.5)",
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        default="mean",
        choices=["mean", "max", "sum"],
        help="SAGEConv aggregation scheme matching the trained model (default: mean)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_path = Path(args.data_dir) / "features_with_labels"
    edges_path = Path(args.data_dir) / "edges"
    model_path = Path(args.model_path)
    output_image = Path(args.output_image)

    print(f"Loading data from:  {features_path}")
    print(f"Loading model from: {model_path}")

    data = load_processed_data(str(features_path), str(edges_path), device)
    _, val_mask, _ = create_split_masks(data.num_nodes, seed=args.seed)

    model = GraphSAGEFraudDetector(
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden_channels,
        out_channels=1,
        num_layers=3,
        dropout=args.dropout,
        aggregator=args.aggregator,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
        targets = data.y[val_mask].cpu().numpy()

    if len(probs) == 0:
        raise ValueError("Validation split is empty. Check --seed or data size.")

    avg_precision = average_precision_score(targets, probs)
    precision, recall, thresholds = precision_recall_curve(targets, probs)

    if thresholds.size == 0:
        raise ValueError("precision_recall_curve produced no thresholds.")

    # Best threshold = maximise F1 across the curve
    sel_precision = precision[:-1]
    sel_recall = recall[:-1]
    f1_scores = (
        2 * sel_precision * sel_recall / (sel_precision + sel_recall + 1e-12)
    )
    best_idx = int(np.nanargmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_precision = float(sel_precision[best_idx])
    best_recall = float(sel_recall[best_idx])
    best_f1 = float(f1_scores[best_idx])

    print("\nValidation PR curve summary:")
    print(f"  PR-AUC (avg precision) : {avg_precision:.4f}")
    print(f"  Optimal threshold      : {best_threshold:.4f}")
    print(f"  Precision @ threshold  : {best_precision:.4f}")
    print(f"  Recall    @ threshold  : {best_recall:.4f}")
    print(f"  F1        @ threshold  : {best_f1:.4f}")
    print(
        f"\n  Use --threshold {best_threshold:.4f} in graphsage_detector.py "
        f"to apply this threshold during training evaluation."
    )

    plot_precision_recall(
        recall=recall,
        precision=precision,
        threshold=best_threshold,
        precision_at_threshold=best_precision,
        recall_at_threshold=best_recall,
        average_precision=avg_precision,
        f1_at_threshold=best_f1,
        aggregator=args.aggregator,
        image_path=output_image,
    )

    print(f"\nSaved PR curve image to: {output_image}")


if __name__ == "__main__":
    main()

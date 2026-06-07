#!/usr/bin/env python3
"""
False-Negative Audit for Phase 1 GCN Model

Identifies all transactions that are ground-truth ILLICIT (label=1) but were
scored below the classification threshold by the GCN — i.e. fraud cases the
model missed.

Outputs:
  - Console summary table (sorted by GCN probability, highest first)
  - CSV saved to data/processed/false_negatives.csv for downstream use

The CSV can feed directly into the Phase 2 agentic triage system:
    python src/agents/run_triage.py --node-id <tx_id>

Usage:
    python find_false_negatives.py
    python find_false_negatives.py --threshold 0.5 --output results/fn.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.graph_detector import GCNFraudDetector

# Classification threshold used during Phase 1 training evaluation
DEFAULT_THRESHOLD = 0.6331


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)

    feat_files = sorted((data_path / "features_with_labels").glob("part-*.parquet"))
    features_df = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)

    edge_files = sorted((data_path / "edges").glob("part-*.parquet"))
    edges_df = pd.concat([pd.read_parquet(f) for f in edge_files], ignore_index=True)

    return features_df, edges_df


@torch.no_grad()
def run_inference(
    features_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    model_path: str,
    hidden_channels: int = 64,
) -> np.ndarray:
    device = torch.device("cpu")

    tx_ids = features_df["tx_id"].values
    feat_cols = [c for c in features_df.columns if c not in ("tx_id", "label")]
    features = features_df[feat_cols].values.astype(np.float32)

    id_to_idx = {tid: i for i, tid in enumerate(tx_ids)}
    src = edges_df["source_id"].map(id_to_idx)
    dst = edges_df["target_id"].map(id_to_idx)
    valid = src.notna() & dst.notna()
    edge_index = torch.tensor(
        np.vstack([src[valid].astype(int).values, dst[valid].astype(int).values]),
        dtype=torch.long,
    )

    model = GCNFraudDetector(
        in_channels=features.shape[1],
        hidden_channels=hidden_channels,
        out_channels=1,
        num_layers=3,
        dropout=0.0,
    )
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    logits = model(torch.tensor(features, dtype=torch.float32), edge_index)
    return torch.sigmoid(logits).numpy()


def find_false_negatives(
    features_df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """
    Return a DataFrame of all false negatives:
    ground-truth ILLICIT nodes that scored below the classification threshold.
    """
    labels = features_df["label"].values
    tx_ids = features_df["tx_id"].values

    fn_mask = (labels == 1) & (probs < threshold)

    fn_df = pd.DataFrame({
        "tx_id":           tx_ids[fn_mask],
        "true_label":      labels[fn_mask],           # always 1
        "gcn_probability": probs[fn_mask].round(6),
        "missed_by":       (threshold - probs[fn_mask]).round(6),  # gap below threshold
    })

    # Highest probability first — these are closest to being caught and most
    # interesting for the triage system (most likely actually illicit)
    fn_df = fn_df.sort_values("gcn_probability", ascending=False).reset_index(drop=True)
    fn_df.index += 1  # 1-based rank
    fn_df.index.name = "rank"

    return fn_df


def print_summary(fn_df: pd.DataFrame, threshold: float, total_illicit: int) -> None:
    total_fn = len(fn_df)
    recall = 1.0 - (total_fn / total_illicit) if total_illicit else 0.0

    print("\n" + "=" * 62)
    print("  FALSE NEGATIVE AUDIT — Phase 1 GCN Model")
    print("=" * 62)
    print(f"  Classification threshold : {threshold}")
    print(f"  Total ground-truth illicit nodes : {total_illicit:,}")
    print(f"  False negatives (missed fraud)   : {total_fn:,}")
    print(f"  Model recall on illicit class    : {recall:.1%}")
    print("=" * 62)

    if fn_df.empty:
        print("  No false negatives found at this threshold.")
        return

    print("\n  Probability distribution of false negatives:")
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    for p in percentiles:
        val = np.percentile(fn_df["gcn_probability"], p)
        print(f"    p{p:>3}: {val:.4f}")

    print(f"\n  Top 20 highest-probability false negatives")
    print(f"  (closest to threshold={threshold}, best candidates for triage)\n")
    print(
        f"  {'Rank':>4}  {'tx_id':>12}  {'GCN Prob':>9}  {'Missed By':>10}"
    )
    print("  " + "-" * 42)
    for rank, row in fn_df.head(20).iterrows():
        print(
            f"  {rank:>4}  {int(row['tx_id']):>12}  "
            f"{row['gcn_probability']:>9.4f}  {row['missed_by']:>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase 1 GCN for false negatives (missed fraud cases)",
    )
    parser.add_argument("--data-dir",         default="data/processed")
    parser.add_argument("--model-path",       default="models/best_model.pt")
    parser.add_argument("--hidden-channels",  type=int,   default=64)
    parser.add_argument("--threshold",        type=float, default=DEFAULT_THRESHOLD,
                        help=f"Classification threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--output",           default="data/processed/false_negatives.csv",
                        help="CSV output path")
    args = parser.parse_args()

    print("Loading processed data …")
    features_df, edges_df = load_data(args.data_dir)

    print("Running GCN inference …")
    probs = run_inference(
        features_df, edges_df, args.model_path, args.hidden_channels
    )

    total_illicit = int((features_df["label"] == 1).sum())
    fn_df = find_false_negatives(features_df, probs, args.threshold)

    print_summary(fn_df, args.threshold, total_illicit)

    # Save CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fn_df.to_csv(out_path)
    print(f"\n  Full list saved to: {out_path}")
    print(f"  To investigate a node: python src/agents/run_triage.py --node-id <tx_id>\n")


if __name__ == "__main__":
    main()

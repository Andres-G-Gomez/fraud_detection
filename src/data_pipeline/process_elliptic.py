"""
PySpark ETL Pipeline for Elliptic Bitcoin Dataset Processing

This module handles:
1. Loading raw Elliptic dataset CSVs
2. Filtering unknown class transactions
3. Mapping class labels (1 -> Fraud, 2 -> Safe)
4. Joining features with class labels
5. Preparing graph data (nodes and edges) for PyTorch Geometric
6. Saving processed data to Parquet format
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "AML-DataPipeline") -> SparkSession:
    """
    Create and configure a Spark session.
    
    Args:
        app_name: Name of the Spark application
        
    Returns:
        Configured SparkSession
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    logger.info(f"Created SparkSession: {app_name}")
    return spark


def load_raw_data(
    spark: SparkSession,
    data_dir: str
) -> Tuple:
    """
    Load the three raw CSV files from the Elliptic dataset.
    
    Args:
        spark: SparkSession
        data_dir: Path to raw data directory
        
    Returns:
        Tuple of (features_df, classes_df, edgelist_df)
    """
    logger.info(f"Loading raw data from {data_dir}")
    
    # Load features (TX ID + 166 features)
    features_df = spark.read.csv(
        f"{data_dir}/elliptic_txs_features.csv",
        header=False,
        inferSchema=True
    )
    
    # Load class labels (TX ID, class) with header row
    classes_df = spark.read.csv(
        f"{data_dir}/elliptic_txs_classes.csv",
        header=True,
        inferSchema=True
    )
    
    # Load edge list (source_id, target_id) with header row
    edgelist_df = spark.read.csv(
        f"{data_dir}/elliptic_txs_edgelist.csv",
        header=True,
        inferSchema=True
    )
    
    # Rename columns for clarity
    if "tx_id" not in features_df.columns:
        features_df = features_df.withColumnRenamed(features_df.columns[0], "tx_id")
    classes_df = classes_df.withColumnRenamed("txId", "tx_id")
    edgelist_df = edgelist_df.withColumnRenamed("txId1", "source_id") \
                              .withColumnRenamed("txId2", "target_id")
    
    logger.info(f"Features shape: {features_df.count()} rows, {len(features_df.columns)} cols")
    logger.info(f"Classes shape: {classes_df.count()} rows")
    logger.info(f"Edgelist shape: {edgelist_df.count()} rows")
    
    return features_df, classes_df, edgelist_df


def has_java() -> bool:
    """Check whether a Java runtime is available in the environment."""
    if os.getenv("JAVA_HOME"):
        java_home = Path(os.getenv("JAVA_HOME"))
        if java_home.exists():
            return True
    try:
        subprocess.run(["java", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def pandas_fallback_pipeline(raw_data_dir: str, output_dir: str) -> None:
    """Process the dataset using pandas when PySpark is unavailable."""
    logger.info("Running pandas fallback pipeline")
    raw_dir = Path(raw_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    features_path = raw_dir / "elliptic_txs_features.csv"
    classes_path = raw_dir / "elliptic_txs_classes.csv"
    edges_path = raw_dir / "elliptic_txs_edgelist.csv"

    features_df = pd.read_csv(features_path, header=None)
    features_df.columns = ["tx_id"] + [f"feature_{i}" for i in range(features_df.shape[1] - 1)]

    classes_df = pd.read_csv(classes_path, header=0)
    classes_df = classes_df.rename(columns={"txId": "tx_id"})
    classes_df["label"] = classes_df["class"].map({"1": 1, "2": 0, 1: 1, 2: 0})
    classes_df = classes_df.dropna(subset=["label"])

    labeled_features = features_df.merge(
        classes_df[["tx_id", "label"]],
        on="tx_id",
        how="inner"
    )

    edges_df = pd.read_csv(edges_path, header=0)
    edges_df = edges_df.rename(columns={"txId1": "source_id", "txId2": "target_id"})

    valid_ids = set(labeled_features["tx_id"].unique())
    filtered_edges = edges_df[
        edges_df["source_id"].isin(valid_ids) &
        edges_df["target_id"].isin(valid_ids)
    ].copy()

    # Save pandas outputs as Parquet files in partition-like directories
    features_output = output_path / "features_with_labels"
    edges_output = output_path / "edges"
    features_output.mkdir(parents=True, exist_ok=True)
    edges_output.mkdir(parents=True, exist_ok=True)

    features_df = labeled_features.reset_index(drop=True)
    edges_df = filtered_edges.reset_index(drop=True)

    features_df.to_parquet(features_output / "part-00000.parquet", index=False, engine="pyarrow")
    edges_df.to_parquet(edges_output / "part-00000.parquet", index=False, engine="pyarrow")

    logger.info(f"Saved pandas processed features to {features_output}")
    logger.info(f"Saved pandas processed edges to {edges_output}")


def process_features_and_labels(
    features_df,
    classes_df
) -> Tuple:
    """
    Process features and labels:
    - Filter out 'unknown' class transactions
    - Map class 1 -> 1 (Fraud), class 2 -> 0 (Safe)
    - Join features with labels
    
    Args:
        features_df: Spark DataFrame with transaction features
        classes_df: Spark DataFrame with class labels
        
    Returns:
        Tuple of (labeled_features_df, fraud_count, safe_count)
    """
    logger.info("Processing features and labels")
    
    # Filter out unknown class
    classes_labeled = classes_df.filter(classes_df["class"] != "unknown")
    
    # Map class labels: 1->Fraud, 2->Safe (handles both strings and numerics)
    classes_labeled = classes_labeled.withColumn(
        "label",
        when(col("class") == 1, 1)
        .when(col("class") == 2, 0)
        .when(col("class") == "1", 1)
        .when(col("class") == "2", 0)
    ).drop("class")
    
    logger.info(f"After filtering unknowns: {classes_labeled.count()} transactions")
    
    # Join features with labels
    labeled_features = features_df.join(
        classes_labeled,
        on="tx_id",
        how="inner"
    )
    
    # Count class distribution
    fraud_count = labeled_features.filter(col("label") == 1).count()
    safe_count = labeled_features.filter(col("label") == 0).count()
    total = fraud_count + safe_count
    fraud_ratio = (fraud_count / total) * 100
    
    logger.info(f"Final dataset: {total} transactions")
    logger.info(f"  Fraud (1): {fraud_count} ({fraud_ratio:.2f}%)")
    logger.info(f"  Safe (0): {safe_count} ({100-fraud_ratio:.2f}%)")
    
    return labeled_features, fraud_count, safe_count


def process_edges(
    spark: SparkSession,
    edgelist_df,
    valid_tx_ids
) -> None:
    """
    Process and filter edge list to include only valid transaction IDs.
    
    Args:
        spark: SparkSession
        edgelist_df: Spark DataFrame with edges
        valid_tx_ids: Sequence of valid transaction IDs (those with known labels)
        
    Returns:
        Processed edge list DataFrame
    """
    logger.info("Processing edges")
    
    valid_tx_ids_df = spark.createDataFrame(
        [(tx_id,) for tx_id in valid_tx_ids],
        ["tx_id"]
    ).distinct()
    
    filtered_edges = edgelist_df.join(
        valid_tx_ids_df.withColumnRenamed("tx_id", "source_id"),
        on="source_id",
        how="inner"
    ).join(
        valid_tx_ids_df.withColumnRenamed("tx_id", "target_id"),
        on="target_id",
        how="inner"
    )
    
    edge_count = filtered_edges.count()
    logger.info(f"Filtered edges: {edge_count} edges")
    
    return filtered_edges


def save_processed_data(
    labeled_features,
    edgelist,
    output_dir: str
) -> None:
    """
    Save processed data to Parquet format for PyTorch Geometric consumption.
    
    Args:
        labeled_features: Spark DataFrame with features and labels
        edgelist: Spark DataFrame with filtered edges
        output_dir: Output directory path
    """
    logger.info(f"Saving processed data to {output_dir}")
    
    # Save features with labels
    features_path = f"{output_dir}/features_with_labels"
    labeled_features.write.parquet(features_path, mode="overwrite")
    logger.info(f"Saved features to {features_path}")
    
    # Save edges
    edges_path = f"{output_dir}/edges"
    edgelist.write.parquet(edges_path, mode="overwrite")
    logger.info(f"Saved edges to {edges_path}")


def main(
    raw_data_dir: str = "data/raw/elliptic_bitcoin_dataset",
    output_dir: str = "data/processed"
) -> None:
    """
    Main ETL pipeline orchestration.
    
    Args:
        raw_data_dir: Path to raw data directory
        output_dir: Path for processed output
    """
    logger.info("=" * 60)
    logger.info("Starting Elliptic Dataset ETL Pipeline")
    logger.info("=" * 60)
    
    # Try to run with Spark if Java is available; otherwise fall back to pandas
    spark = None
    try:
        if has_java():
            spark = create_spark_session()
            features_df, classes_df, edgelist_df = load_raw_data(spark, raw_data_dir)
            labeled_features, fraud_count, safe_count = process_features_and_labels(
                features_df, classes_df
            )
            valid_tx_ids = labeled_features.select("tx_id").rdd.map(lambda x: x[0]).collect()
            filtered_edges = process_edges(spark, edgelist_df, valid_tx_ids)
            save_processed_data(labeled_features, filtered_edges, output_dir)
        else:
            logger.warning("Java runtime not detected; using pandas fallback pipeline.")
            pandas_fallback_pipeline(raw_data_dir, output_dir)

        logger.info("=" * 60)
        logger.info("ETL Pipeline completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"ETL Pipeline failed with error: {str(e)}", exc_info=True)
        raise

    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()

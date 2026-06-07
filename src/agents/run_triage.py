#!/usr/bin/env python3
"""
Phase 2: Agentic Triage System for Bitcoin Transaction Fraud Detection

Initialises a CrewAI multi-agent pipeline (Gemini backend) that automatically
investigates a transaction node flagged as high-risk by the Phase 1 GCN model.

Three agents run sequentially:
  1. Graph Data Gatherer    – retrieves node features and 1-hop neighbours
  2. AML Policy Analyst    – evaluates against three injected AML typologies
  3. Compliance Officer    – makes a final structured routing decision

Usage:
    python src/agents/run_triage.py [--node-id NODE_ID] [--data-dir DIR]
                                    [--model-path PATH] [--hidden-channels N]

Environment variables:
    GEMINI_API_KEY  (required) – Google Gemini API key
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Make project root importable regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root (no-op if file is absent, so CI/prod env vars still work)
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PYDANTIC OUTPUT MODEL  (enforces Agent 3's structured response)
# ─────────────────────────────────────────────────────────────────────────────

class TriageDecision(BaseModel):
    """Structured compliance routing decision produced by the Triage Officer."""

    triage_decision: Literal["FILE_SAR", "DISMISS", "ESCALATE_TO_HUMAN"] = Field(
        description="Final compliance routing decision."
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Analyst confidence in the decision (0.0 = low, 1.0 = high).",
    )
    justification: str = Field(
        description="2-3 sentence explanation citing specific evidence from the analysis."
    )
    sar_report_markdown: Optional[str] = Field(
        default=None,
        description=(
            "Full Suspicious Activity Report in Markdown if decision is FILE_SAR; "
            "null otherwise."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODULE-LEVEL DATA STORE  (populated once by _initialise_graph_data)
# ─────────────────────────────────────────────────────────────────────────────

_features_df: Optional[pd.DataFrame] = None
_edges_df: Optional[pd.DataFrame] = None
_tx_id_to_label: dict = {}
_fraud_probs: Optional[np.ndarray] = None
_tx_ids: Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING & GCN INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def _load_processed_data(data_dir: str) -> tuple:
    data_path = Path(data_dir)

    feat_files = sorted((data_path / "features_with_labels").glob("part-*.parquet"))
    if not feat_files:
        raise FileNotFoundError(
            f"No feature parquet files found in {data_path / 'features_with_labels'}"
        )
    features_df = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)

    edge_files = sorted((data_path / "edges").glob("part-*.parquet"))
    if not edge_files:
        raise FileNotFoundError(
            f"No edge parquet files found in {data_path / 'edges'}"
        )
    edges_df = pd.concat([pd.read_parquet(f) for f in edge_files], ignore_index=True)

    logger.info(f"Loaded {len(features_df):,} nodes and {len(edges_df):,} edges")
    return features_df, edges_df


@torch.no_grad()
def _run_gcn_inference(
    features_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    model_path: str,
    hidden_channels: int = 64,
) -> np.ndarray:
    """Load the Phase 1 GCN and compute per-node fraud probabilities."""
    from src.models.graph_detector import GCNFraudDetector

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
        dropout=0.0,  # No stochastic dropout during deterministic inference
    )
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    logits = model(torch.tensor(features, dtype=torch.float32), edge_index)
    probs = torch.sigmoid(logits).numpy()

    logger.info(
        f"GCN inference complete — score range [{probs.min():.4f}, {probs.max():.4f}]"
    )
    return probs


def _find_top_risk_node(
    features_df: pd.DataFrame, fraud_probs: np.ndarray
) -> tuple:
    """Return (node_id_str, probability) of the highest-scoring illicit node."""
    tx_ids = features_df["tx_id"].values
    labels = features_df["label"].values
    # Prefer nodes that are both ground-truth illicit and GCN high-confidence
    ranked = np.where(labels == 1, fraud_probs, -1.0)
    top_idx = int(np.argmax(ranked))
    node_id = str(tx_ids[top_idx])
    prob = float(fraud_probs[top_idx])
    logger.info(f"Auto-selected top-risk node: {node_id}  GCN p={prob:.4f}")
    return node_id, prob


def _initialise_graph_data(
    data_dir: str, model_path: str, hidden_channels: int = 64
) -> None:
    """Populate module-level stores. Must be called before running the Crew."""
    global _features_df, _edges_df, _tx_id_to_label, _fraud_probs, _tx_ids

    _features_df, _edges_df = _load_processed_data(data_dir)
    _tx_id_to_label = dict(
        zip(_features_df["tx_id"], _features_df["label"].astype(int))
    )
    _tx_ids = _features_df["tx_id"].values

    if Path(model_path).exists():
        _fraud_probs = _run_gcn_inference(
            _features_df, _edges_df, model_path, hidden_channels
        )
    else:
        logger.warning(f"Model not found at '{model_path}'. GCN scores unavailable.")
        _fraud_probs = None


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GRAPH QUERY LOGIC  (called by Agent 1's tool)
# ─────────────────────────────────────────────────────────────────────────────

def _query_node(node_id_str: str) -> str:
    """
    Return a human-readable intelligence report for a single transaction node.
    Covers: ground-truth label, GCN fraud probability, and 1-hop graph topology.
    """
    global _features_df, _edges_df, _tx_id_to_label, _fraud_probs, _tx_ids

    if _features_df is None or _edges_df is None:
        return "ERROR: Graph data not initialised. Run _initialise_graph_data() first."

    # Normalise ID type (dataset uses integers)
    try:
        node_id = int(str(node_id_str).strip())
    except (ValueError, AttributeError):
        node_id = str(node_id_str).strip()

    row = _features_df[_features_df["tx_id"] == node_id]
    if row.empty:
        return (
            f"ERROR: Node '{node_id}' not found in the labeled Elliptic dataset. "
            "Verify the node ID is from the processed parquet files."
        )

    true_label = int(row["label"].iloc[0])
    true_label_str = "ILLICIT (1)" if true_label == 1 else "LICIT (0)"

    # GCN fraud probability
    fraud_prob: Optional[float] = None
    if _fraud_probs is not None and _tx_ids is not None:
        match = np.where(_tx_ids == node_id)[0]
        if len(match):
            fraud_prob = float(_fraud_probs[match[0]])

    # 1-hop neighbourhood
    inbound_ids = _edges_df[_edges_df["target_id"] == node_id]["source_id"].unique()
    outbound_ids = _edges_df[_edges_df["source_id"] == node_id]["target_id"].unique()

    def classify_neighbors(ids):
        lines = []
        for nid in ids[:25]:
            lbl = _tx_id_to_label.get(nid)
            tag = "ILLICIT" if lbl == 1 else ("LICIT" if lbl == 0 else "UNKNOWN")
            lines.append(f"  - txId {nid}: {tag}")
        if len(ids) > 25:
            lines.append(f"  ... and {len(ids) - 25} additional neighbours (truncated)")
        return lines or ["  (none)"]

    illicit_in = sum(1 for n in inbound_ids if _tx_id_to_label.get(n) == 1)
    illicit_out = sum(1 for n in outbound_ids if _tx_id_to_label.get(n) == 1)

    # Feature statistics
    feat_cols = [c for c in _features_df.columns if c not in ("tx_id", "label")]
    feat = row[feat_cols].values[0].astype(float)
    fan_in = len(inbound_ids)
    fan_out = len(outbound_ids)

    report_lines = [
        "=" * 62,
        "  TRANSACTION GRAPH INTELLIGENCE REPORT",
        "=" * 62,
        f"  Node ID (txId)         : {node_id}",
        f"  Ground-Truth Label     : {true_label_str}",
        f"  GCN Fraud Probability  : {f'{fraud_prob:.4f}' if fraud_prob is not None else 'N/A'}",
        "",
        "  ── INBOUND CONNECTIONS (this node RECEIVED funds FROM) ──",
        f"  Distinct inbound nodes : {fan_in}",
        f"    of which ILLICIT     : {illicit_in}",
        f"    of which LICIT       : {fan_in - illicit_in - sum(1 for n in inbound_ids if _tx_id_to_label.get(n) is None)}",
        "  Node detail:",
        *classify_neighbors(inbound_ids),
        "",
        "  ── OUTBOUND CONNECTIONS (this node SENT funds TO) ──",
        f"  Distinct outbound nodes: {fan_out}",
        f"    of which ILLICIT     : {illicit_out}",
        f"    of which LICIT       : {fan_out - illicit_out - sum(1 for n in outbound_ids if _tx_id_to_label.get(n) is None)}",
        "  Node detail:",
        *classify_neighbors(outbound_ids),
        "",
        "  ── TOPOLOGY SUMMARY ──",
        f"  Fan-in  (unique inbound nodes)  : {fan_in}",
        f"  Fan-out (unique outbound nodes) : {fan_out}",
        f"  Fan-in / Fan-out ratio          : {fan_in / max(fan_out, 1):.2f}",
        f"  Total illicit 1-hop neighbours  : {illicit_in + illicit_out}  "
        f"({illicit_in} inbound, {illicit_out} outbound)",
        "",
        "  ── NODE FEATURE STATISTICS ──",
        f"  Feature vector length  : {len(feat)}",
        f"  Value range            : [{feat.min():.4f}, {feat.max():.4f}]",
        f"  Mean / Std             : {feat.mean():.4f} / {feat.std():.4f}",
        "=" * 62,
    ]
    return "\n".join(report_lines)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CREWAI CREW CONSTRUCTION & EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def _build_crew_tool():
    """Create the CrewAI BaseTool that Agent 1 uses to query the graph."""
    from crewai.tools import BaseTool
    from pydantic import BaseModel as _Base

    class _Input(_Base):
        node_id: str

    class GraphQueryTool(BaseTool):
        name: str = "graph_query_tool"
        description: str = (
            "Queries the Bitcoin transaction graph for a given node ID. "
            "Returns a complete intelligence report: ground-truth AML label, "
            "GCN fraud probability from Phase 1, 1-hop inbound/outbound neighbours "
            "with their ILLICIT/LICIT/UNKNOWN labels, fan-in/fan-out topology stats, "
            "and node feature statistics. "
            "Input — node_id: the transaction ID string."
        )
        args_schema: type[_Base] = _Input

        def _run(self, node_id: str) -> str:  # noqa: D401
            return _query_node(node_id)

    return GraphQueryTool()


def run_triage_crew(node_id: str, fraud_probability: float) -> TriageDecision:
    """Build and execute the sequential 3-agent CrewAI crew. Returns TriageDecision."""
    from crewai import Agent, Task, Crew, Process, LLM

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Add it to your .env file or export it before running this script."
        )

    llm = LLM(
        model="gemini/gemini-2.5-flash",
        temperature=0.1,
        api_key=api_key,
        max_retries=5,
    )

    graph_tool = _build_crew_tool()

    # ── Agent 1: Graph Data Gatherer ─────────────────────────────────────────
    gatherer = Agent(
        role="Data Retrieval Specialist",
        goal=(
            "Retrieve complete, accurate transaction data and graph topology "
            "for the flagged Bitcoin transaction node."
        ),
        backstory=(
            "You are an expert in blockchain data systems. Your sole responsibility "
            "is to query the graph database and present raw data faithfully and "
            "completely. You never interpret or draw conclusions—you only retrieve "
            "and format data for the downstream compliance team."
        ),
        tools=[graph_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # ── Agent 2: AML Policy Analyst ──────────────────────────────────────────
    # TECHNIQUE: System Prompt / Backstory Injection
    # All three AML typology definitions are hard-coded into the backstory so the
    # agent applies them consistently without needing external tools.
    analyst = Agent(
        role="Financial Crime Typology Expert",
        goal=(
            "Evaluate the retrieved transaction data against established AML "
            "typologies and produce a rigorous, evidence-based policy analysis."
        ),
        backstory=(
            "You are a Senior Financial Crime Expert with 15 years at a Tier-1 "
            "investment bank's AML Compliance unit, certified CAMS. You have "
            "authored internal typology playbooks and trained regulators.\n\n"

            "You ALWAYS evaluate flagged transactions against exactly THREE typologies:\n\n"

            "== TYPOLOGY 1 -- SMURFING (Structuring) ==\n"
            "Definition: A node receives funds from MANY distinct source nodes and "
            "forwards them to a SINGLE (or very few) destination nodes. This "
            "aggregation pattern is designed to break large illicit sums into "
            "smaller transfers that individually fall below reporting thresholds.\n"
            "Key Indicators:\n"
            "  • High fan-in (many distinct inbound nodes)\n"
            "  • Low fan-out (few distinct outbound nodes)\n"
            "  • Fan-in / Fan-out ratio significantly greater than 3.0\n\n"

            "== TYPOLOGY 2 -- PASS-THROUGH (Layering) ==\n"
            "Definition: A node acts purely as a relay intermediary. Inbound "
            "transaction count closely mirrors outbound count, indicating the node "
            "adds no economic activity — it merely obscures the money trail.\n"
            "Key Indicators:\n"
            "  • Fan-in / Fan-out ratio close to 1.0 (roughly equal)\n"
            "  • Both inbound AND outbound connections exist\n"
            "  • Presence of illicit nodes on either side heightens risk\n\n"

            "== TYPOLOGY 3 -- DARK MARKET CONNECTION ==\n"
            "Definition: The flagged node has at least one direct 1-hop edge to a "
            "confirmed illicit node as identified by the Phase 1 GCN model labels.\n"
            "Key Indicators:\n"
            "  • illicit_inbound_count > 0  (received funds from illicit node)\n"
            "  • illicit_outbound_count > 0  (sent funds to illicit node)\n"
            "  • Even a single confirmed illicit neighbour satisfies this typology\n\n"

            "For each typology state: TRIGGERED or NOT TRIGGERED with specific "
            "numeric evidence. Do not speculate beyond the data provided."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # ── Agent 3: Compliance Triage Officer ───────────────────────────────────
    officer = Agent(
        role="Final Decision Maker",
        goal=(
            "Produce a final, legally defensible compliance routing decision "
            "in strict structured JSON format based on the Policy Analyst's report."
        ),
        backstory=(
            "You are the Chief Compliance Officer. Your decisions are final and "
            "legally defensible. You apply the following decision framework:\n"
            "  FILE_SAR            — 2+ typologies triggered, OR Dark Market "
            "Connection confirmed, OR GCN fraud probability > 0.85\n"
            "  ESCALATE_TO_HUMAN  — Exactly 1 typology triggered and GCN score "
            "between 0.50 and 0.85, or evidence is genuinely ambiguous\n"
            "  DISMISS            — Zero typologies triggered and GCN score < 0.50\n\n"
            "You output ONLY valid JSON. No markdown fences, no prose outside the JSON."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # ── Task 1 ────────────────────────────────────────────────────────────────
    task_gather = Task(
        description=(
            f"A Bitcoin transaction node has been flagged by the Phase 1 GCN model.\n\n"
            f"  Node ID             : {node_id}\n"
            f"  GCN Fraud Probability: {fraud_probability:.4f}\n\n"
            f"Call the graph_query_tool with this node_id to retrieve its complete "
            f"data: ground-truth label, GCN probability, all 1-hop neighbours with "
            f"their ILLICIT/LICIT labels, and full topology statistics. "
            f"Present the raw report without interpretation."
        ),
        expected_output=(
            "The verbatim output of the graph_query_tool: a structured intelligence "
            "report containing the node's label, GCN fraud score, inbound/outbound "
            "neighbour lists with labels, fan-in/fan-out counts, illicit neighbour "
            "totals, and feature statistics."
        ),
        agent=gatherer,
        tools=[graph_tool],
    )

    # ── Task 2 ────────────────────────────────────────────────────────────────
    task_analyse = Task(
        description=(
            "You have received the transaction graph intelligence report from the "
            "Data Retrieval Specialist. Apply your three AML typology framework:\n\n"
            "For EACH of the three typologies produce:\n"
            "  Status   : TRIGGERED or NOT TRIGGERED\n"
            "  Evidence : Specific numeric data points from the report\n"
            "  Risk     : HIGH / MEDIUM / LOW\n\n"
            "Close with an Overall Risk Summary (2-3 sentences) that clearly states "
            "which typologies fired and what the aggregate risk level is."
        ),
        expected_output=(
            "A structured typology analysis with three clearly labelled sections "
            "(Smurfing, Pass-Through, Dark Market Connection), each with a "
            "TRIGGERED/NOT TRIGGERED verdict, numeric evidence, and risk rating, "
            "followed by an Overall Risk Summary paragraph."
        ),
        agent=analyst,
        context=[task_gather],
    )

    # ── Task 3 ────────────────────────────────────────────────────────────────
    task_decide = Task(
        description=(
            "You have read the AML Policy Analyst's typology report. "
            "Apply your decision framework and output a SINGLE valid JSON object "
            "with EXACTLY these four fields — no other text:\n\n"
            '{\n'
            '  "triage_decision"     : "FILE_SAR" | "DISMISS" | "ESCALATE_TO_HUMAN",\n'
            '  "confidence_score"    : <float 0.0–1.0>,\n'
            '  "justification"       : "<2-3 sentence explanation with evidence>",\n'
            '  "sar_report_markdown" : "<full SAR in Markdown if FILE_SAR, else null>"\n'
            '}\n\n'
            "If triage_decision is FILE_SAR the SAR must include: "
            "Report Date, Subject Transaction ID, Reporting Entity, "
            "Synopsis, Typologies Identified, Supporting Evidence, and "
            "Recommended Action."
        ),
        expected_output=(
            "A single valid JSON object conforming to the TriageDecision schema: "
            "triage_decision (FILE_SAR | DISMISS | ESCALATE_TO_HUMAN), "
            "confidence_score (float 0.0–1.0), justification (2-3 sentences), "
            "sar_report_markdown (Markdown SAR string or null)."
        ),
        agent=officer,
        context=[task_analyse],
        output_pydantic=TriageDecision,
    )

    # ── Assemble and kick off ─────────────────────────────────────────────────
    crew = Crew(
        agents=[gatherer, analyst, officer],
        tasks=[task_gather, task_analyse, task_decide],
        process=Process.sequential,
        verbose=True,
    )

    logger.info(
        f"Kicking off triage crew — node {node_id}, GCN p={fraud_probability:.4f}"
    )
    result = crew.kickoff()

    # ── Extract TriageDecision from crew output ───────────────────────────────
    decision: Optional[TriageDecision] = None

    # Primary: crew-level pydantic attribute (CrewAI ≥ 0.80)
    if hasattr(result, "pydantic") and isinstance(result.pydantic, TriageDecision):
        decision = result.pydantic

    # Fallback: last task output's pydantic attribute
    if decision is None and hasattr(result, "tasks_output") and result.tasks_output:
        last = result.tasks_output[-1]
        if hasattr(last, "pydantic") and isinstance(last.pydantic, TriageDecision):
            decision = last.pydantic

    # Fallback: regex-extract JSON from raw string output and parse manually
    if decision is None:
        raw = str(result)
        try:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                data = json.loads(match.group())
                decision = TriageDecision(**data)
        except Exception as exc:
            logger.error(f"JSON fallback parsing failed: {exc}")
            logger.debug(f"Raw crew output:\n{raw}")

    if decision is None:
        raise RuntimeError(
            "Could not extract a valid TriageDecision from the crew output. "
            "Check the verbose logs above for the raw agent responses."
        )

    return decision


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 Agentic Triage — Bitcoin Fraud Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--node-id",
        default=None,
        help=(
            "Transaction node ID to investigate. "
            "Auto-selects the highest-risk illicit node if omitted."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Path to Phase 1 processed data directory (default: data/processed)",
    )
    parser.add_argument(
        "--model-path",
        default="models/best_model.pt",
        help="Path to saved GCN weights (default: models/best_model.pt)",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="GCN hidden channel count — must match the trained model (default: 64)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    banner = "  PHASE 2: AGENTIC TRIAGE SYSTEM — Bitcoin Fraud Detection  "
    print("\n" + "=" * len(banner))
    print(banner)
    print("=" * len(banner) + "\n")

    # Step 1 – load Phase 1 data and run GCN inference
    print("[1/3] Loading processed data and running Phase 1 GCN inference …")
    _initialise_graph_data(
        data_dir=args.data_dir,
        model_path=args.model_path,
        hidden_channels=args.hidden_channels,
    )

    # Step 2 – resolve target node
    if args.node_id:
        node_id = args.node_id
        fraud_prob: float = 0.0
        if _fraud_probs is not None and _tx_ids is not None:
            try:
                typed = int(node_id)
            except ValueError:
                typed = node_id
            idx = np.where(_tx_ids == typed)[0]
            if len(idx):
                fraud_prob = float(_fraud_probs[idx[0]])
        print(
            f"[2/3] Using specified node: {node_id}  "
            f"(GCN fraud probability: {fraud_prob:.4f})"
        )
    else:
        print("[2/3] Auto-selecting highest-risk node from Phase 1 results …")
        if _fraud_probs is None or _features_df is None:
            raise RuntimeError(
                "GCN inference failed and no --node-id was provided. "
                "Check that the model file exists."
            )
        node_id, fraud_prob = _find_top_risk_node(_features_df, _fraud_probs)
        print(
            f"      → Node: {node_id}  |  GCN fraud probability: {fraud_prob:.4f}"
        )

    # Step 3 – run the CrewAI triage pipeline
    print(f"\n[3/3] Launching CrewAI triage crew for node {node_id} …\n")
    decision = run_triage_crew(node_id=node_id, fraud_probability=fraud_prob)

    # Print the final structured output
    separator = "=" * 62
    print(f"\n{separator}")
    print("  FINAL TRIAGE DECISION  (Structured Pydantic Output)")
    print(separator)
    print(decision.model_dump_json(indent=2))
    print(f"{separator}\n")

    # Exit code encodes the decision for downstream automation
    exit_code = {"FILE_SAR": 2, "ESCALATE_TO_HUMAN": 1, "DISMISS": 0}.get(
        decision.triage_decision, 0
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

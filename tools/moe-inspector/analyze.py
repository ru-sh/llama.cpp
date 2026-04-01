#!/usr/bin/env python3
"""
MoE expert invocation distribution analysis.

Reads routing records produced by llama-moe-inspector and generates:
  out/heatmap_code.png      — expert × layer selection frequency, code domain
  out/heatmap_nl.png        — same for NL domain
  out/heatmap_diff.png      — diverging difference (code − nl)
  out/jsd_by_layer.png      — per-layer Jensen-Shannon divergence
  out/unused_experts.csv    — experts selected < UNUSED_THRESHOLD of the time
  out/report.md             — summary with embedded plot links

Usage:
  python3 analyze.py --code data/code_routing.jsonl --nl data/nl_routing.jsonl --out out/

  # Override model dimensions if needed:
  python3 analyze.py --code ... --nl ... --n-expert 128 --n-expert-used 8 --n-layers 94
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import jensenshannon

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

UNUSED_THRESHOLD = 0.001   # expert selected < 0.1 % of tokens → "unused"

# -------------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"warning: skipping malformed line {lineno} in {path}: {e}", file=sys.stderr)
    return records


def infer_dims(records: list[dict]) -> tuple[int, int]:
    """Return (n_expert_from_data, n_expert_used_from_data) by scanning records."""
    max_expert = 0
    expert_used_counts = set()
    for rec in records:
        expert_used_counts.add(len(rec["experts"]))
        for eid in rec["experts"]:
            max_expert = max(max_expert, eid)
    n_expert      = max_expert + 1
    n_expert_used = max(expert_used_counts) if expert_used_counts else 1
    return n_expert, n_expert_used


# -------------------------------------------------------------------------
# Core computation
# -------------------------------------------------------------------------

def build_frequency_matrix(
    records: list[dict],
    n_layers: int,
    n_expert: int,
) -> np.ndarray:
    """
    Returns hist[layer, expert] = selection count (not yet normalised).
    Layers and experts beyond the declared dimensions are silently ignored.
    """
    hist = np.zeros((n_layers, n_expert), dtype=np.int64)
    skipped = 0
    for rec in records:
        layer = rec["layer"]
        if layer < 0 or layer >= n_layers:
            skipped += 1
            continue
        for eid in rec["experts"]:
            if 0 <= eid < n_expert:
                hist[layer, eid] += 1
    if skipped:
        print(f"  warning: skipped {skipped} records with out-of-range layer index", file=sys.stderr)
    return hist


def normalise_rows(hist: np.ndarray) -> np.ndarray:
    """Convert counts to probability distributions per layer (row)."""
    row_sums = hist.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    return hist.astype(np.float64) / safe_sums


def compute_jsd_per_layer(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Jensen-Shannon divergence for each layer.
    jensenshannon returns the square root of JSD by default; we square it
    to get the actual divergence in [0, 1].
    """
    n_layers = p.shape[0]
    jsd = np.zeros(n_layers)
    for layer in range(n_layers):
        jsd[layer] = jensenshannon(p[layer], q[layer]) ** 2
    return jsd


def find_unused_experts(
    prob: np.ndarray,
    domain: str,
    threshold: float = UNUSED_THRESHOLD,
) -> list[dict]:
    """Return list of {domain, layer, expert_id, freq} for rarely-used experts."""
    unused = []
    for layer in range(prob.shape[0]):
        for expert in range(prob.shape[1]):
            if prob[layer, expert] < threshold:
                unused.append({
                    "domain":    domain,
                    "layer":     layer,
                    "expert_id": expert,
                    "freq":      float(prob[layer, expert]),
                })
    return unused


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_heatmap(
    prob: np.ndarray,
    title: str,
    out_path: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        prob.T,              # axes: x=layer, y=expert
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="Selection frequency")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Expert ID")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_jsd(jsd: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    layers = np.arange(len(jsd))
    ax.plot(layers, jsd, linewidth=1.5, color="steelblue")
    ax.axhline(0.1, linestyle="--", linewidth=0.8, color="tomato", label="JSD = 0.1")
    ax.fill_between(layers, jsd, alpha=0.2, color="steelblue")
    ax.set_xlabel("Layer")
    ax.set_ylabel("JSD(code ∥ NL)")
    ax.set_title("Per-layer Jensen-Shannon Divergence: code vs NL")
    ax.set_ylim(0, max(jsd.max() * 1.1, 0.15))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------

def write_report(
    out_dir: Path,
    n_tokens_code: int,
    n_tokens_nl: int,
    n_expert: int,
    n_layers: int,
    jsd: np.ndarray,
    prob_code: np.ndarray,
    prob_nl: np.ndarray,
    unused_code: list[dict],
    unused_nl: list[dict],
) -> None:
    pct_unused_code = 100.0 * len(unused_code) / (n_layers * n_expert)
    pct_unused_nl   = 100.0 * len(unused_nl)   / (n_layers * n_expert)

    # Top-5 code-biased experts (highest mean freq in code vs NL)
    diff_mean = prob_code.mean(axis=0) - prob_nl.mean(axis=0)
    top_code_experts = np.argsort(diff_mean)[-5:][::-1].tolist()
    top_nl_experts   = np.argsort(diff_mean)[:5].tolist()

    report_path = out_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("# MoE Expert Invocation Distribution Report\n\n")

        f.write("## Summary\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Total tokens — code  | {n_tokens_code:,} |\n")
        f.write(f"| Total tokens — NL    | {n_tokens_nl:,} |\n")
        f.write(f"| Number of layers     | {n_layers} |\n")
        f.write(f"| Number of experts    | {n_expert} |\n")
        f.write(f"| Unused experts (code) | {len(unused_code)} ({pct_unused_code:.1f}%) |\n")
        f.write(f"| Unused experts (NL)   | {len(unused_nl)} ({pct_unused_nl:.1f}%) |\n")
        f.write(f"| Mean JSD across layers | {jsd.mean():.4f} |\n")
        f.write(f"| Max  JSD (layer {int(jsd.argmax())}) | {jsd.max():.4f} |\n\n")

        f.write("## Per-Layer JSD Table\n\n")
        f.write("Layers with JSD > 0.1 are marked **bold**.\n\n")
        f.write("| Layer | JSD | >\u20090.1? |\n|---|---|---|\n")
        for layer in range(n_layers):
            flag = "**yes**" if jsd[layer] > 0.1 else ""
            f.write(f"| {layer} | {jsd[layer]:.4f} | {flag} |\n")

        f.write("\n## Top-5 Code-Biased Experts (higher freq in code than NL)\n\n")
        f.write("| Expert ID | Mean freq code | Mean freq NL | Δ |\n|---|---|---|---|\n")
        for eid in top_code_experts:
            f.write(f"| {eid} | {prob_code[:,eid].mean():.5f} | "
                    f"{prob_nl[:,eid].mean():.5f} | "
                    f"{diff_mean[eid]:+.5f} |\n")

        f.write("\n## Top-5 NL-Biased Experts (higher freq in NL than code)\n\n")
        f.write("| Expert ID | Mean freq code | Mean freq NL | Δ |\n|---|---|---|---|\n")
        for eid in top_nl_experts:
            f.write(f"| {eid} | {prob_code[:,eid].mean():.5f} | "
                    f"{prob_nl[:,eid].mean():.5f} | "
                    f"{diff_mean[eid]:+.5f} |\n")

        f.write("\n## Plots\n\n")
        f.write("![code heatmap](heatmap_code.png)\n\n")
        f.write("![nl heatmap](heatmap_nl.png)\n\n")
        f.write("![diff heatmap](heatmap_diff.png)\n\n")
        f.write("![JSD by layer](jsd_by_layer.png)\n\n")

    print(f"  saved {report_path}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyse MoE routing JSONL files.")
    ap.add_argument("--code",         required=True, help="JSONL routing file for code domain")
    ap.add_argument("--nl",           required=True, help="JSONL routing file for NL domain")
    ap.add_argument("--out",          default="out", help="output directory (default: out)")
    ap.add_argument("--n-expert",     type=int, default=0,
                    help="total experts (0 = infer from data)")
    ap.add_argument("--n-expert-used",type=int, default=0,
                    help="experts per token (0 = infer from data)")
    ap.add_argument("--n-layers",     type=int, default=0,
                    help="number of MoE layers (0 = infer from data)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading routing records...")
    records_code = load_jsonl(args.code)
    records_nl   = load_jsonl(args.nl)

    if not records_code:
        sys.exit(f"error: no records loaded from {args.code}")
    if not records_nl:
        sys.exit(f"error: no records loaded from {args.nl}")

    print(f"  code records : {len(records_code):,}")
    print(f"  nl   records : {len(records_nl):,}")

    # Infer or use provided dims
    n_expert_code,      n_exp_used_code = infer_dims(records_code)
    n_expert_nl,        n_exp_used_nl   = infer_dims(records_nl)
    n_expert_data  = max(n_expert_code, n_expert_nl)
    n_exp_used     = max(n_exp_used_code, n_exp_used_nl)

    n_expert  = args.n_expert   or n_expert_data
    n_layers  = args.n_layers   or (max(r["layer"] for r in records_code + records_nl) + 1)
    if args.n_expert_used:
        n_exp_used = args.n_expert_used

    print(f"\nDimensions: n_layers={n_layers}  n_expert={n_expert}  n_expert_used={n_exp_used}")

    # Count unique tokens (per position) per domain
    n_tokens_code = len({(r["pos"],) for r in records_code})
    n_tokens_nl   = len({(r["pos"],) for r in records_nl})

    # Build frequency matrices
    print("\nBuilding frequency matrices...")
    hist_code = build_frequency_matrix(records_code, n_layers, n_expert)
    hist_nl   = build_frequency_matrix(records_nl,   n_layers, n_expert)

    prob_code = normalise_rows(hist_code)
    prob_nl   = normalise_rows(hist_nl)

    # JSD per layer
    print("Computing per-layer JSD...")
    jsd = compute_jsd_per_layer(prob_code, prob_nl)

    # Unused experts
    unused_code = find_unused_experts(prob_code, "code")
    unused_nl   = find_unused_experts(prob_nl,   "nl")

    # Write unused_experts.csv
    csv_path = out_dir / "unused_experts.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "layer", "expert_id", "freq"])
        writer.writeheader()
        writer.writerows(unused_code)
        writer.writerows(unused_nl)
    print(f"\nSaved {csv_path}  ({len(unused_code)+len(unused_nl)} rows)")

    # Heatmaps
    print("\nGenerating plots...")
    vmax_common = max(prob_code.max(), prob_nl.max())

    plot_heatmap(
        prob_code,
        title="Expert selection frequency — code domain",
        out_path=str(out_dir / "heatmap_code.png"),
        cmap="viridis", vmin=0, vmax=vmax_common,
    )
    plot_heatmap(
        prob_nl,
        title="Expert selection frequency — NL domain",
        out_path=str(out_dir / "heatmap_nl.png"),
        cmap="viridis", vmin=0, vmax=vmax_common,
    )

    diff = prob_code - prob_nl
    abs_max = np.abs(diff).max()
    plot_heatmap(
        diff,
        title="Expert frequency difference (code − NL)",
        out_path=str(out_dir / "heatmap_diff.png"),
        cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
    )

    plot_jsd(jsd, str(out_dir / "jsd_by_layer.png"))

    # Report
    print("\nWriting report...")
    write_report(
        out_dir,
        n_tokens_code=n_tokens_code,
        n_tokens_nl=n_tokens_nl,
        n_expert=n_expert,
        n_layers=n_layers,
        jsd=jsd,
        prob_code=prob_code,
        prob_nl=prob_nl,
        unused_code=unused_code,
        unused_nl=unused_nl,
    )

    # Verify all output files exist and are non-empty (DoD check)
    required = [
        out_dir / "heatmap_code.png",
        out_dir / "heatmap_nl.png",
        out_dir / "heatmap_diff.png",
        out_dir / "jsd_by_layer.png",
        out_dir / "unused_experts.csv",
        out_dir / "report.md",
    ]
    missing = [p for p in required if not p.exists() or p.stat().st_size == 0]
    if missing:
        sys.exit(f"error: the following outputs are missing or empty: {missing}")

    print("\nAll outputs verified. Done.")
    print(f"\nMean JSD : {jsd.mean():.4f}")
    print(f"Max  JSD : {jsd.max():.4f}  (layer {int(jsd.argmax())})")
    print(f"Unused experts (code) : {len(unused_code)} / {n_layers * n_expert}")
    print(f"Unused experts (nl)   : {len(unused_nl)}   / {n_layers * n_expert}")


if __name__ == "__main__":
    main()

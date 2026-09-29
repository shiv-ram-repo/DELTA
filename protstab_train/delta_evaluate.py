"""
delta_evaluate.py — DELTA Evaluation + Decomposition Analysis
==============================================================

Evaluates DELTA on all benchmark datasets and produces decomposition
analyses for the paper:

1. Standard metrics: Spearman, Pearson, RMSE, MAE on all 10+ benchmarks
2. Gain-loss decomposition scatter (Figure: gain vs loss colored by ΔΔG)
3. Mutation-type heatmap (Figure: 20×20 average gain, loss, ΔΔG per substitution)
4. Ssym forward/reverse analysis (how well does the decomposition handle anti-symmetry)
5. Per-mutation-type performance breakdown

Usage:
    python -m protstab_train.delta_evaluate \\
        --data_root /path/to/data \\
        --ckpt runs/delta_v1/best.pt \\
        --out runs/delta_v1/eval
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from protstab_data import (
    Alphabet,
    Featurizer,
    MegaScaleTestDatasets,
    MegaScaleDataset,
)
from protstab_models.delta import DELTA, DELTAConfig
from protstab_models.decomposition import DecompositionConfig

from .metrics import compute_metrics

log = logging.getLogger(__name__)

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _move_batch(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def _onehot_to_idx(onehot):
    return onehot.argmax(dim=-1).clamp(max=19)


# ═══════════════════════════════════════════════════════════════════════
# Core evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_delta(
    model: DELTA,
    loader,
    device,
    name: str = "",
) -> dict:
    """
    Evaluate DELTA on a dataset, returning predictions + decomposition.

    Returns dict with:
        preds, targets, gains, losses: numpy arrays
        wt_aa, mut_aa: lists of amino acid characters
        metrics: dict of standard metrics
    """
    model.eval()
    all_preds, all_targets = [], []
    all_gains, all_losses = [], []
    all_wt_aa, all_mut_aa = [], []

    for batch in tqdm(loader, desc=f"Eval {name}", leave=False):
        batch = _move_batch(batch, device)

        ddg, gain, loss_scores = model(batch, return_decomposition=True)

        # Collect predictions
        all_preds.append(ddg.view(-1).cpu().numpy())
        all_targets.append(batch["ddG"].view(-1).cpu().numpy())
        all_gains.append(gain.view(-1).cpu().numpy())
        all_losses.append(loss_scores.view(-1).cpu().numpy())

        # Extract amino acid identities
        at = batch["append_tensors"]
        wt_idx = _onehot_to_idx(at[:, :21].float()).cpu().numpy()
        mut_idx = _onehot_to_idx(at[:, 21:].float()).cpu().numpy()
        all_wt_aa.extend([ALPHABET[i] for i in wt_idx])
        all_mut_aa.extend([ALPHABET[i] for i in mut_idx])

    if not all_preds:
        return {
            "preds": np.array([]), "targets": np.array([]),
            "gains": np.array([]), "losses": np.array([]),
            "wt_aa": [], "mut_aa": [], "metrics": {},
        }

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    gains = np.concatenate(all_gains)
    losses = np.concatenate(all_losses)
    metrics = compute_metrics(preds, targets)

    return {
        "preds": preds, "targets": targets,
        "gains": gains, "losses": losses,
        "wt_aa": all_wt_aa, "mut_aa": all_mut_aa,
        "metrics": metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# Decomposition Analysis
# ═══════════════════════════════════════════════════════════════════════

def build_mutation_type_heatmaps(results: dict) -> dict:
    """
    Build 20×20 heatmaps of average gain, loss, and ΔΔG per substitution type.

    Returns dict with:
        gain_map, loss_map, ddg_map: (20, 20) numpy arrays
        count_map: (20, 20) count of samples per cell
    """
    gain_map = np.zeros((20, 20))
    loss_map = np.zeros((20, 20))
    ddg_map = np.zeros((20, 20))
    count_map = np.zeros((20, 20))

    for i, (wt, mut) in enumerate(zip(results["wt_aa"], results["mut_aa"])):
        if wt not in ALPHABET or mut not in ALPHABET:
            continue
        wi = ALPHABET.index(wt)
        mi = ALPHABET.index(mut)

        gain_map[wi, mi] += results["gains"][i]
        loss_map[wi, mi] += results["losses"][i]
        ddg_map[wi, mi] += results["preds"][i]
        count_map[wi, mi] += 1

    # Average (avoid division by zero)
    mask = count_map > 0
    gain_map[mask] /= count_map[mask]
    loss_map[mask] /= count_map[mask]
    ddg_map[mask] /= count_map[mask]

    return {
        "gain_map": gain_map,
        "loss_map": loss_map,
        "ddg_map": ddg_map,
        "count_map": count_map,
    }


def compute_per_mutation_type_metrics(results: dict) -> pd.DataFrame:
    """
    Compute Spearman correlation per mutation type (e.g., A→V, L→P).

    Returns DataFrame with columns:
        wt, mut, n, spearman, pearson, mean_gain, mean_loss, mean_ddg
    """
    rows = []
    wt_arr = np.array(results["wt_aa"])
    mut_arr = np.array(results["mut_aa"])

    for wt in ALPHABET:
        for mut in ALPHABET:
            if wt == mut:
                continue
            mask = (wt_arr == wt) & (mut_arr == mut)
            n = mask.sum()
            if n < 5:
                continue

            p = results["preds"][mask]
            t = results["targets"][mask]
            g = results["gains"][mask]
            l = results["losses"][mask]
            m = compute_metrics(p, t)

            rows.append({
                "wt": wt, "mut": mut, "n": int(n),
                "spearman": m["spearman"], "pearson": m["pearson"],
                "rmse": m["rmse"],
                "mean_gain": float(np.mean(g)),
                "mean_loss": float(np.mean(l)),
                "mean_ddg_pred": float(np.mean(p)),
                "mean_ddg_true": float(np.mean(t)),
            })

    return pd.DataFrame(rows).sort_values("n", ascending=False)


def analyze_antisymmetry(results_fwd: dict, results_rev: dict) -> dict:
    """
    Analyze forward/reverse consistency of the decomposition.

    For Ssym-style datasets where we have both A→B and B→A mutations,
    checks:
      1. ΔΔG(A→B) + ΔΔG(B→A) ≈ 0
      2. gain_fwd(B) ≈ loss_rev(B) (the gain of B in forward should
         match what B loses in reverse)
    """
    ddg_fwd = results_fwd["preds"]
    ddg_rev = results_rev["preds"]
    n = min(len(ddg_fwd), len(ddg_rev))

    if n == 0:
        return {"n": 0}

    ddg_fwd = ddg_fwd[:n]
    ddg_rev = ddg_rev[:n]

    anti_sum = ddg_fwd + ddg_rev
    return {
        "n": n,
        "mean_antisym_error": float(np.mean(np.abs(anti_sum))),
        "std_antisym_error": float(np.std(anti_sum)),
        "max_antisym_error": float(np.max(np.abs(anti_sum))),
        "median_antisym_error": float(np.median(np.abs(anti_sum))),
    }


# ═══════════════════════════════════════════════════════════════════════
# Full benchmark evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_all_benchmarks(
    data_root: str,
    ckpt_path: str,
    output_dir: str,
    *,
    proteinmpnn_ckpt: str = "",
    esm_name: str = "esm2_t33_650M_UR50D",
    adapter_layer: int = -3,
    tune_mpnn: bool = True,
    device: Optional[str] = None,
    # DELTA-specific
    d_aa: int = 64,
    d_sub: int = 64,
    d_hidden: int = 512,
    n_branch_layers: int = 3,
    k_neighbors: int = 32,
    bidirectional: bool = False,
) -> pd.DataFrame:
    """
    Evaluate DELTA checkpoint on every test benchmark.
    Produces standard metrics + decomposition analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("DELTA Evaluation")
    print("=" * 80)

    # Build model
    decomp_cfg = DecompositionConfig(
        d_model=1408,
        d_aa=d_aa, d_sub=d_sub, d_hidden=d_hidden,
        n_branch_layers=n_branch_layers, k_neighbors=k_neighbors,
    )

    delta_cfg = DELTAConfig()
    delta_cfg.adapter_layer_indices = [adapter_layer]
    delta_cfg.name = esm_name
    delta_cfg.proteinmpnn_ckpt = proteinmpnn_ckpt
    delta_cfg.encoder.tune = tune_mpnn
    delta_cfg.mlp.input_dim = 128
    delta_cfg.mlp.hidden_dim = [512, 512, 512]
    delta_cfg.mlp.output_dim = 21
    delta_cfg.mlp.dropout = 0.1
    delta_cfg.decomposition = decomp_cfg
    delta_cfg.bidirectional = bidirectional

    model = DELTA(delta_cfg).to(device)

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:
        state = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    model.eval()

    # Data
    alphabet = Alphabet(name="esm")
    featurizer = Featurizer(alphabet)
    collection = MegaScaleTestDatasets(data_root=data_root)

    # Run benchmarks
    all_results = []
    all_metrics = []

    print("\n" + "=" * 80)
    print("Running benchmarks")
    print("=" * 80)

    for name, dataset in collection.iter_named():
        print(f"\nDataset: {name}")
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=2, collate_fn=featurizer,
        )

        results = evaluate_delta(model, loader, device, name=name)
        metrics = results["metrics"]
        metrics["dataset"] = name
        all_metrics.append(metrics)
        all_results.append((name, results))

        print(
            f"  ρ={metrics['spearman']:.4f} | r={metrics['pearson']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | "
            f"|gain|={np.mean(np.abs(results['gains'])):.3f} | "
            f"|loss|={np.mean(np.abs(results['losses'])):.3f}"
        )

        # Save per-dataset predictions
        np.save(os.path.join(output_dir, f"preds_{name}.npy"), results["preds"])
        np.save(os.path.join(output_dir, f"targets_{name}.npy"), results["targets"])
        np.save(os.path.join(output_dir, f"gains_{name}.npy"), results["gains"])
        np.save(os.path.join(output_dir, f"losses_{name}.npy"), results["losses"])

    # Summary table
    df = pd.DataFrame(all_metrics)[
        ["dataset", "n", "spearman", "pearson", "r2", "rmse", "mae"]
    ]
    csv_path = os.path.join(output_dir, "test_metrics.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nSaved metrics to: {csv_path}")

    # ── Decomposition Analysis ────────────────────────────────────
    print("\n" + "=" * 80)
    print("DECOMPOSITION ANALYSIS")
    print("=" * 80)

    # Use Megascale test for the main decomposition analysis
    for name, results in all_results:
        if "megascale" in name.lower() or "test" in name.lower():
            print(f"\nMutation-type analysis on: {name}")

            # Heatmaps
            heatmaps = build_mutation_type_heatmaps(results)
            np.save(os.path.join(output_dir, f"gain_heatmap_{name}.npy"), heatmaps["gain_map"])
            np.save(os.path.join(output_dir, f"loss_heatmap_{name}.npy"), heatmaps["loss_map"])
            np.save(os.path.join(output_dir, f"ddg_heatmap_{name}.npy"), heatmaps["ddg_map"])
            np.save(os.path.join(output_dir, f"count_heatmap_{name}.npy"), heatmaps["count_map"])

            # Per-mutation-type metrics
            type_metrics = compute_per_mutation_type_metrics(results)
            type_csv = os.path.join(output_dir, f"mutation_type_metrics_{name}.csv")
            type_metrics.to_csv(type_csv, index=False, float_format="%.4f")
            print(f"  Saved mutation-type metrics to: {type_csv}")

            # Top/bottom substitution types
            if len(type_metrics) > 0:
                print(f"\n  Best substitution types (by Spearman):")
                top = type_metrics.nlargest(5, "spearman")
                for _, row in top.iterrows():
                    print(f"    {row['wt']}→{row['mut']}: ρ={row['spearman']:.3f} (n={row['n']})")

                print(f"\n  Worst substitution types (by Spearman):")
                bot = type_metrics.nsmallest(5, "spearman")
                for _, row in bot.iterrows():
                    print(f"    {row['wt']}→{row['mut']}: ρ={row['spearman']:.3f} (n={row['n']})")
            break  # Only do detailed analysis for first matching dataset

    # Extract learned mutation-type embedding for visualization
    embed = model.decomp_head.mut_type_embed.embed.detach().cpu().numpy()
    np.save(os.path.join(output_dir, "learned_mut_type_embedding.npy"), embed)
    print(f"\nSaved learned mutation-type embedding: {embed.shape}")

    # Extract learned distance kernel scales
    sigma = model.decomp_head.context_encoder.log_sigma.exp().detach().cpu().numpy()
    print(f"Learned distance scales (per head): {sigma}")
    np.save(os.path.join(output_dir, "learned_distance_scales.npy"), sigma)

    return df


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    p = argparse.ArgumentParser(description="DELTA Evaluation")
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mpnn_ckpt", default="")
    p.add_argument("--esm_name", default="esm2_t33_650M_UR50D")
    p.add_argument("--d_sub", type=int, default=64)
    p.add_argument("--d_hidden", type=int, default=512)
    p.add_argument("--k_neighbors", type=int, default=32)
    p.add_argument("--bidirectional", action="store_true",
                   help="Load a DELTA-B (bidirectional) checkpoint")
    args = p.parse_args()

    evaluate_all_benchmarks(
        data_root=args.data_root,
        ckpt_path=args.ckpt,
        output_dir=args.out,
        proteinmpnn_ckpt=args.mpnn_ckpt,
        esm_name=args.esm_name,
        d_sub=args.d_sub,
        d_hidden=args.d_hidden,
        k_neighbors=args.k_neighbors,
        bidirectional=args.bidirectional,
    )

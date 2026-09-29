"""
run_delta_ablation.py — DELTA Ablation Sweep
==============================================

Runs the ablation configurations needed for the APBC paper:

  Config A: Baseline SPURS (existing repo, no DELTA changes)
  Config B: DELTA decomposition only (no anti-symmetry, no mutation-type embed)
  Config C: DELTA + mutation-type conditioning (BLOSUM init)
  Config D: DELTA + mutation-type + BCAS anti-symmetry
  Config E: DELTA + mutation-type + BCAS + decomposition anti-symmetry
  Config F: DELTA full (E + decomposition regularizer)  ← headline
  Config G: DELTA full + BMC loss (to test BMC interaction)

Each configuration is run with 3 seeds for variance estimation.
Results are saved to a summary CSV for the paper table.

Usage:
    python run_delta_ablation.py --data_root /path/to/data --output_base runs/ablation

Expected wall time: ~6-8 hours per config on A6000 (× 7 configs × 3 seeds = ~126-168 GPU-hours)
"""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from protstab_train.delta_train import DeltaTrainConfig, train

log = logging.getLogger(__name__)


ABLATION_CONFIGS = {
    # Config A: will be evaluated from existing SPURS checkpoint (not trained here)
    # Run the existing training pipeline instead.

    "B_decomp_only": dict(
        use_blosum_init=False,
        use_bcas=False,
        use_decomp_antisym=False,
        use_decomp_reg=False,
        use_mut_smoothness=False,
    ),
    "C_mut_type": dict(
        use_blosum_init=True,
        use_bcas=False,
        use_decomp_antisym=False,
        use_decomp_reg=False,
    ),
    "D_mut_type_bcas": dict(
        use_blosum_init=True,
        use_bcas=True,
        use_decomp_antisym=False,
        use_decomp_reg=False,
    ),
    "E_mut_type_bcas_dasym": dict(
        use_blosum_init=True,
        use_bcas=True,
        use_decomp_antisym=True,
        use_decomp_reg=False,
    ),
    "F_full": dict(
        use_blosum_init=True,
        use_bcas=True,
        use_decomp_antisym=True,
        use_decomp_reg=True,
        decomp_reg_weight=0.1,
    ),
    "G_full_bmc": dict(
        use_blosum_init=True,
        use_bcas=True,
        use_decomp_antisym=True,
        use_decomp_reg=True,
        loss_type="bmc",
    ),
}

SEEDS = [42, 123, 456]


def run_ablation(
    data_root: str,
    output_base: str,
    proteinmpnn_ckpt: str = "",
    max_epochs: int = 200,
    configs: dict = None,
    seeds: list = None,
    backbone_ckpt: str = "",
    backbone_lr_scale: float = 0.1,
):
    """Run the full ablation sweep."""
    if configs is None:
        configs = ABLATION_CONFIGS
    if seeds is None:
        seeds = SEEDS

    all_results = []

    for config_name, overrides in configs.items():
        for seed in seeds:
            run_name = f"{config_name}_seed{seed}"
            run_dir = os.path.join(output_base, run_name)

            log.info(f"\n{'='*70}")
            log.info(f"Running: {run_name}")
            log.info(f"{'='*70}")

            # Build config
            cfg = DeltaTrainConfig(
                data_root=data_root,
                output_dir=run_dir,
                proteinmpnn_ckpt=proteinmpnn_ckpt,
                max_epochs=max_epochs,
                seed=seed,
                backbone_ckpt=backbone_ckpt,
                backbone_lr_scale=backbone_lr_scale,
                **overrides,
            )

            # Train
            try:
                ckpt_path = train(cfg)
                log.info(f"Completed: {run_name} → {ckpt_path}")
            except Exception as e:
                log.error(f"Failed: {run_name} → {e}")
                continue

            # Read best validation metric from training log
            log_path = os.path.join(run_dir, "training_log.csv")
            if os.path.exists(log_path):
                df = pd.read_csv(log_path)
                best_row = df.loc[df["val_spearman"].idxmax()]
                all_results.append({
                    "config": config_name,
                    "seed": seed,
                    "best_epoch": int(best_row["epoch"]),
                    "val_spearman": float(best_row["val_spearman"]),
                    "val_pearson": float(best_row["val_pearson"]),
                    "val_rmse": float(best_row["val_rmse"]),
                    "ckpt_path": ckpt_path,
                })

    # Summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary = results_df.groupby("config").agg(
            mean_spearman=("val_spearman", "mean"),
            std_spearman=("val_spearman", "std"),
            mean_pearson=("val_pearson", "mean"),
            std_pearson=("val_pearson", "std"),
            mean_rmse=("val_rmse", "mean"),
            n_seeds=("seed", "count"),
        ).round(4)

        summary_path = os.path.join(output_base, "ablation_summary.csv")
        summary.to_csv(summary_path)
        results_df.to_csv(
            os.path.join(output_base, "ablation_all_runs.csv"), index=False,
        )

        print("\n" + "=" * 70)
        print("ABLATION SUMMARY")
        print("=" * 70)
        print(summary.to_string())
        print(f"\nSaved to: {summary_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    p = argparse.ArgumentParser(description="DELTA Ablation Sweep")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_base", default="runs/delta_ablation")
    p.add_argument("--mpnn_ckpt", default="")
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--backbone_ckpt", default="",
                   help="Pretrained backbone checkpoint (match headline DELTA-U)")
    p.add_argument("--backbone_lr_scale", type=float, default=0.1)
    p.add_argument("--configs", nargs="+", default=None,
                   help="Subset of config names to run (default: all)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Seeds to use (default: 42 123 456)")
    args = p.parse_args()

    configs = ABLATION_CONFIGS
    if args.configs:
        configs = {k: v for k, v in ABLATION_CONFIGS.items() if k in args.configs}

    run_ablation(
        data_root=args.data_root,
        output_base=args.output_base,
        proteinmpnn_ckpt=args.mpnn_ckpt,
        max_epochs=args.max_epochs,
        configs=configs,
        seeds=args.seeds or SEEDS,
        backbone_ckpt=args.backbone_ckpt,
        backbone_lr_scale=args.backbone_lr_scale,
    )

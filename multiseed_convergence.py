"""
multiseed_convergence.py — Do g/l Converge Across Seeds?
=========================================================

Addresses R2 and R3: the identifiability worry has a direct empirical test —
if the decomposition is a stable property of the data+objective (rather than
an arbitrary per-run solution), then independently trained seeds should
produce CONSISTENT per-mutation gain/loss values, not just consistent net
ddG.

Given decomposition CSVs from >=2 seeds (same dataset), this script:
  1. aligns mutations across seeds by (protein, position, wt, mut)
  2. reports how reproducible the NET prediction is (upper bound: it is
     supervised) vs the COMPONENTS (the real question)
  3. reports cross-seed Spearman and Pearson of gain and of loss
  4. reports the diagnostic scores (burial, reverse-consistency) per seed
     with mean +- std, giving the confidence intervals reviewers requested

Interpretation:
  - High cross-seed component correlation  -> the decomposition the model
    finds is largely determined by data+objective, mitigating the
    non-identifiability concern empirically.
  - Low component correlation but stable diagnostics -> the SPECIFIC values
    vary but their STRUCTURE (burial ordering) is stable; report honestly.

Usage:
    python multiseed_convergence.py \\
        --csvs diag_seed42/decomposition_s2648.csv \\
               diag_seed123/decomposition_s2648.csv \\
               diag_seed456/decomposition_s2648.csv \\
        --out multiseed_out
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

KEY = ["protein", "position", "wt", "mut"]


def diag_scores(df):
    m = df["contacts"] > 0
    rl, _ = spearmanr(df.loc[m, "loss_fwd"].abs(), df.loc[m, "contacts"])
    rg, _ = spearmanr(df.loc[m, "gain_fwd"].abs(), df.loc[m, "contacts"])
    rev = float("nan")
    if "loss_rev" in df.columns:
        rev, _ = spearmanr(df["gain_fwd"], df["loss_rev"])
    return rl, rg, rev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--out", default="multiseed_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seeds = []
    for i, path in enumerate(args.csvs):
        if not os.path.exists(path):
            print(f"  missing: {path}"); continue
        d = pd.read_csv(path)
        if not set(KEY).issubset(d.columns):
            raise SystemExit(f"{path} lacks alignment keys {KEY}. "
                             "Re-run diagnose_decomposition.py (records protein+position).")
        seeds.append((f"seed{i}", d))
    if len(seeds) < 2:
        raise SystemExit("Need >=2 seed CSVs.")

    print("=" * 70)
    print("MULTI-SEED CONVERGENCE")
    print("=" * 70)

    # Per-seed diagnostics -> mean +- std (the CIs reviewers asked for)
    rows = []
    for name, d in seeds:
        rl, rg, rev = diag_scores(d)
        rows.append({"seed": name, "r_loss_burial": rl, "r_gain_burial": rg,
                     "reverse_consistency": rev})
    sdf = pd.DataFrame(rows)
    print("\nPer-seed diagnostics:")
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nMean +- std across seeds:")
    for col in ["r_loss_burial", "r_gain_burial", "reverse_consistency"]:
        print(f"  {col:22s} {sdf[col].mean():.3f} +- {sdf[col].std():.3f}")
    sdf.to_csv(os.path.join(args.out, "per_seed_diagnostics.csv"), index=False)

    # Cross-seed component agreement on shared mutations
    print("\n" + "-" * 70)
    print("Cross-seed agreement on shared mutations (aligned by protein+pos+wt+mut):")
    pair_rows = []
    for (n1, d1), (n2, d2) in itertools.combinations(seeds, 2):
        m = d1.merge(d2, on=KEY, suffixes=("_1", "_2"))
        if len(m) < 10:
            print(f"  {n1} vs {n2}: only {len(m)} shared, skipping"); continue
        # net ddg reproducibility (supervised -> upper bound)
        r_ddg, _ = spearmanr(m["ddg_fwd_1"], m["ddg_fwd_2"])
        # component reproducibility (the real question)
        r_gain_s, _ = spearmanr(m["gain_fwd_1"], m["gain_fwd_2"])
        r_loss_s, _ = spearmanr(m["loss_fwd_1"], m["loss_fwd_2"])
        r_gain_p, _ = pearsonr(m["gain_fwd_1"], m["gain_fwd_2"])
        r_loss_p, _ = pearsonr(m["loss_fwd_1"], m["loss_fwd_2"])
        pair_rows.append({
            "pair": f"{n1}-{n2}", "n_shared": len(m),
            "rho_ddg": round(r_ddg, 3),
            "rho_gain": round(r_gain_s, 3), "rho_loss": round(r_loss_s, 3),
            "r_gain": round(r_gain_p, 3), "r_loss": round(r_loss_p, 3),
        })
        print(f"  {n1} vs {n2} (n={len(m)}): "
              f"ddG rho={r_ddg:.3f} | gain rho={r_gain_s:.3f} "
              f"loss rho={r_loss_s:.3f}")
    pd.DataFrame(pair_rows).to_csv(
        os.path.join(args.out, "cross_seed_agreement.csv"), index=False)

    print("""
  Reading the result:
   - rho_ddg is high by construction (net is supervised); it is the ceiling.
   - rho_gain / rho_loss near rho_ddg  -> components are reproducible; the
     decomposition is largely determined by data+objective (identifiability
     concern is empirically mitigated).
   - rho_gain / rho_loss much lower than rho_ddg  -> the specific split varies
     across seeds even though structure-level diagnostics (above) are stable;
     report this honestly as the components being structurally but not
     pointwise identifiable.
""")


if __name__ == "__main__":
    main()

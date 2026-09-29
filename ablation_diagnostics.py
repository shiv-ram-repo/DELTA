"""
ablation_diagnostics.py — Diagnostics Across Ablation Configs (R1, R2)
======================================================================

Reviewers R1 and R2 ask the key methodological question: how much of the
reported interpretability is DIRECTLY INDUCED by the imposed priors
(anti-symmetry L_bcas, decomposition anti-symmetry L_dec, mutation-type
conditioning, magnitude regularizer), versus emergent?

run_delta_ablation.py trains the ablation configs (A..G, removing terms one
at a time). This script runs the SAME diagnostic suite on each ablation
checkpoint and assembles a comparison table, so the paper can state exactly
which diagnostics survive when each prior is removed.

The crucial columns:
  - reverse-consistency: EXPECTED to drop sharply when L_dec/L_bcas removed
    (reviewers correctly note D1 is partly enforced by these). Showing the
    drop is the honest move — it quantifies the enforcement.
  - burial association: if this SURVIVES removal of the anti-symmetry terms,
    it is the genuinely emergent diagnostic (not enforced), which is the
    strongest thing the paper can claim.

This script calls diagnose_decomposition.collect_decomposition on each
checkpoint. It requires the same environment as diagnose_decomposition.py.

Usage:
    python ablation_diagnostics.py \\
        --data_root /path/to/data \\
        --ablation_dir runs/delta_ablation \\
        --dataset s2648 \\
        --out ablation_diag_out

It auto-discovers checkpoints named best.pt under <ablation_dir>/<config>/<seed>/.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from protstab_data import Alphabet, Featurizer, MegaScaleTestDatasets
from diagnose_decomposition import load_model, collect_decomposition

CONSERVATIVE = {("L","I"),("I","V"),("V","I"),("D","E"),("E","D"),("K","R"),
                ("R","K"),("F","Y"),("Y","F"),("S","T"),("L","M"),("M","L"),
                ("I","L"),("V","L"),("L","V")}


def scores_from_df(df):
    out = {}
    # reverse consistency
    if {"gain_fwd", "loss_rev"}.issubset(df.columns):
        out["reverse_consistency"], _ = spearmanr(df["gain_fwd"], df["loss_rev"])
    # conservative ratio
    isc = df.apply(lambda r: (r["wt"], r["mut"]) in CONSERVATIVE, axis=1)
    cm = (df[isc]["gain_fwd"].abs() + df[isc]["loss_fwd"].abs()).mean()
    nm = (df[~isc]["gain_fwd"].abs() + df[~isc]["loss_fwd"].abs()).mean()
    out["conservative_ratio"] = cm / nm if nm else float("nan")
    # burial
    m = df["contacts"] > 0
    out["burial_loss"], _ = spearmanr(df.loc[m, "loss_fwd"].abs(), df.loc[m, "contacts"])
    out["burial_gain"], _ = spearmanr(df.loc[m, "gain_fwd"].abs(), df.loc[m, "contacts"])
    out["burial_gap"] = out["burial_loss"] - out["burial_gain"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ablation_dir", required=True)
    ap.add_argument("--dataset", default="s2648")
    ap.add_argument("--out", default="ablation_diag_out")
    ap.add_argument("--d_sub", type=int, default=64)
    ap.add_argument("--d_hidden", type=int, default=512)
    ap.add_argument("--k_neighbors", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # discover checkpoints: <ablation_dir>/<config>[/<seed>]/best.pt
    ckpts = sorted(glob.glob(os.path.join(args.ablation_dir, "**", "best.pt"), recursive=True))
    if not ckpts:
        raise SystemExit(f"No best.pt found under {args.ablation_dir}")
    print(f"Found {len(ckpts)} checkpoints")

    # data (loaded once)
    alphabet = Alphabet(name="esm")
    featurizer = Featurizer(alphabet)
    collection = MegaScaleTestDatasets(data_root=args.data_root)
    loader = None
    for name, ds in collection.iter_named():
        if name == args.dataset:
            loader = DataLoader(ds, batch_size=1, shuffle=False,
                                num_workers=2, collate_fn=featurizer)
            break
    if loader is None:
        raise SystemExit(f"dataset {args.dataset} not found")

    rows = []
    for ck in ckpts:
        # config label = parent dir names between ablation_dir and best.pt
        rel = os.path.relpath(ck, args.ablation_dir)
        label = os.path.dirname(rel).replace(os.sep, "/")
        is_bidir = "bidir" in label.lower() or "delta_b" in label.lower()
        print(f"\n[{label}] loading...")
        try:
            model = load_model(ck, args.data_root, device, bidirectional=is_bidir,
                               d_sub=args.d_sub, d_hidden=args.d_hidden,
                               k_neighbors=args.k_neighbors)
            df = collect_decomposition(model, loader, device)
            sc = scores_from_df(df)
            sc["config"] = label
            sc["n"] = len(df)
            rows.append(sc)
            print(f"  reverse={sc.get('reverse_consistency', float('nan')):.3f}  "
                  f"cons_ratio={sc['conservative_ratio']:.3f}  "
                  f"burial_gap={sc['burial_gap']:+.3f}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append({"config": label, "error": str(e)})

    out_df = pd.DataFrame(rows)
    cols = ["config", "n", "reverse_consistency", "conservative_ratio",
            "burial_loss", "burial_gain", "burial_gap"]
    out_df = out_df[[c for c in cols if c in out_df.columns] +
                    [c for c in out_df.columns if c not in cols]]
    out_df.to_csv(os.path.join(args.out, "ablation_diagnostics.csv"), index=False)

    print("\n" + "=" * 70)
    print("ABLATION DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {args.out}/ablation_diagnostics.csv")
    print("""
  Camera-ready reading:
   - If reverse_consistency COLLAPSES when L_dec/L_bcas are removed, state
     plainly that D1 is largely enforced by those priors (reviewers R2/R3
     are correct on this point) — do not claim D1 as emergent.
   - If burial_gap SURVIVES removal of the anti-symmetry priors, that is the
     genuinely emergent diagnostic; foreground it as the load-bearing
     evidence and de-emphasize D1 as independent support.
""")


if __name__ == "__main__":
    main()

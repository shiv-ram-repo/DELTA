"""
build_paper_numbers.py — Consolidate Every Number the Paper Needs
==================================================================

Reads the eval CSVs and diagnostic outputs from all DELTA runs (and the
SPURS baseline) and prints a single consolidated report organized to match
the \\VERIFY{} placeholders in the manuscript. This turns filling the paper
from "hunt through 6 directories" into "copy from one printout."

It also computes seed mean±std where multiple seed runs are provided, so
you report the Ssym-direct spread honestly instead of cherry-picking.

Usage:
    python build_paper_numbers.py \\
        --delta_u_eval runs/delta_pretrained/eval/test_metrics.csv \\
        --delta_u_seeds runs/delta_full/eval/test_metrics.csv runs/delta_seed123/eval/test_metrics.csv \\
        --delta_b_eval runs/delta_bidir/eval/test_metrics.csv \\
        --spurs_eval runs/spurs_baseline_eval/spurs_test_metrics.csv \\
        --diag_u diag_s2648/... \\
        --out paper_numbers.txt

All arguments are optional; the script reports whatever it is given.
For diagnostics, point --diag_u / --diag_b at the decomposition CSVs
(decomposition_*.csv) produced by diagnose_decomposition.py, and the
script recomputes the headline diagnostic numbers from them.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
CONSERVATIVE = {("L","I"),("I","V"),("V","I"),("D","E"),("E","D"),("K","R"),
                ("R","K"),("F","Y"),("Y","F"),("S","T"),("L","M"),("M","L"),
                ("I","L"),("V","L"),("L","V")}


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def report_accuracy_table(paths_dict):
    """paths_dict: {label: csv_path}. Builds the main comparison table."""
    section("TABLE 1 — Spearman ρ by benchmark (paste into \\VERIFY cells)")
    frames = {}
    for label, path in paths_dict.items():
        if path and os.path.exists(path):
            df = pd.read_csv(path).set_index("dataset")
            frames[label] = df["spearman"]
    if not frames:
        print("  (no eval CSVs provided)")
        return
    table = pd.DataFrame(frames)
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))


def report_seed_spread(seed_paths):
    """Report mean±std across seed runs — critical for honest Ssym reporting."""
    section("SEED MEAN ± STD (report these, do NOT cherry-pick best seed)")
    if not seed_paths:
        print("  (no seed CSVs provided)")
        return
    frames = []
    for path in seed_paths:
        if os.path.exists(path):
            frames.append(pd.read_csv(path).set_index("dataset")["spearman"])
    if not frames:
        print("  (none found)")
        return
    combined = pd.concat(frames, axis=1)
    mean = combined.mean(axis=1)
    std = combined.std(axis=1)
    for ds in combined.index:
        vals = ", ".join(f"{v:.4f}" for v in combined.loc[ds].values)
        print(f"  {ds:18s} mean={mean[ds]:.4f}  std={std[ds]:.4f}   (seeds: {vals})")


def recompute_diagnostics(decomp_csv, label):
    """Recompute the headline diagnostic numbers from a decomposition CSV."""
    section(f"DIAGNOSTICS — {label}  ({os.path.basename(decomp_csv)})")
    if not (decomp_csv and os.path.exists(decomp_csv)):
        print("  (CSV not provided/found)")
        return
    df = pd.read_csv(decomp_csv)

    # Reverse-consistency
    if {"gain_fwd","loss_rev","loss_fwd","gain_rev"}.issubset(df.columns):
        r1, _ = spearmanr(df["gain_fwd"], df["loss_rev"])
        r2, _ = spearmanr(df["loss_fwd"], df["gain_rev"])
        print(f"  Reverse-consistency: ρ(gain_fwd,loss_rev)={r1:.3f}  ρ(loss_fwd,gain_rev)={r2:.3f}")

    # Conservative attenuation
    if {"wt","mut","gain_fwd","loss_fwd"}.issubset(df.columns):
        is_cons = df.apply(lambda r: (r["wt"], r["mut"]) in CONSERVATIVE, axis=1)
        cons_mag = (df[is_cons]["gain_fwd"].abs() + df[is_cons]["loss_fwd"].abs()).mean()
        nonc_mag = (df[~is_cons]["gain_fwd"].abs() + df[~is_cons]["loss_fwd"].abs()).mean()
        if nonc_mag and not np.isnan(cons_mag):
            print(f"  Conservative attenuation: cons={cons_mag:.3f}  noncons={nonc_mag:.3f}  "
                  f"ratio={cons_mag/nonc_mag:.3f}  (n_cons={is_cons.sum()})")

    # Burial association
    if {"loss_fwd","gain_fwd","contacts"}.issubset(df.columns):
        valid = df[df["contacts"] > 0]
        rl, _ = spearmanr(valid["loss_fwd"].abs(), valid["contacts"])
        rg, _ = spearmanr(valid["gain_fwd"].abs(), valid["contacts"])
        print(f"  Burial association: ρ(|loss|,contacts)={rl:.3f}  ρ(|gain|,contacts)={rg:.3f}")

    # Apportionment of destabilization
    if {"gain_fwd","loss_fwd","ddg_true"}.issubset(df.columns):
        destab = df[df["ddg_true"] > 1.0]
        if len(destab):
            g = destab["gain_fwd"].mean()
            l = -destab["loss_fwd"].mean()
            frac = g / (g + l + 1e-9)
            print(f"  Destabilization apportionment: gain fraction={frac:.1%}  (n={len(destab)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delta_u_eval", default="")
    p.add_argument("--delta_b_eval", default="")
    p.add_argument("--spurs_eval", default="")
    p.add_argument("--delta_u_seeds", nargs="*", default=[])
    p.add_argument("--diag_u", default="", help="DELTA-U decomposition CSV")
    p.add_argument("--diag_b", default="", help="DELTA-B decomposition CSV")
    p.add_argument("--out", default="paper_numbers.txt")
    args = p.parse_args()

    # Tee output to file
    class Tee:
        def __init__(self, *fs): self.fs = fs
        def write(self, x):
            for f in self.fs: f.write(x)
        def flush(self):
            for f in self.fs: f.flush()
    fh = open(args.out, "w")
    sys.stdout = Tee(sys.__stdout__, fh)

    print("DELTA — Consolidated Paper Numbers")
    print("Generated for filling manuscript \\VERIFY{} placeholders.")

    report_accuracy_table({
        "DELTA-U": args.delta_u_eval,
        "DELTA-B": args.delta_b_eval,
        "SPURS":   args.spurs_eval,
    })
    report_seed_spread(args.delta_u_seeds)
    recompute_diagnostics(args.diag_u, "DELTA-U")
    recompute_diagnostics(args.diag_b, "DELTA-B")

    section("CHECKLIST")
    print("""  [ ] Table 1 cells filled from the accuracy table above
  [ ] Ssym-direct reported as mean±std (see seed spread)
  [ ] Reverse-consistency ρ stated WITH its dataset name
  [ ] Conservative ratio stated WITH dataset
  [ ] Burial ρ stated as 'modest' (loss vs gain both ~0.3)
  [ ] DELTA-B apportionment + anti-symmetry contrast filled
  [ ] SPURS column present (run eval_spurs_baseline.py if missing)""")

    sys.stdout = sys.__stdout__
    fh.close()
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()

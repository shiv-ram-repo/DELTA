"""
case_study_lookup.py — Per-Mutation Case Studies (Burial-Conditional)
======================================================================

Extracts SPECIFIC mutations (not pair-averages) with their DELTA gain/loss
components AND their structural burial, so the case-study figure can be
built on the burial-conditional reading that the analysis actually supports
— NOT on raw branch magnitude, which does not support a clean "loss > gain
for cavity mutations" claim.

The honest case-study narrative this enables:
    "At buried, high-contact wild-type positions, the loss component is
     amplified, consistent with the documented structural role of core
     packing in these mutations."
NOT:
    "The loss component exceeds the gain component for cavity mutations."
    (Your data does not support this; do not write it.)

This script ranks individual mutations by wild-type burial and reports the
decomposition for the most-buried and least-buried examples, so you can
pick well-characterized cases for a figure and show the burial→loss trend
within real proteins.

Usage:
    # Needs the decomposition CSV WITH a 'contacts' column and, ideally, a
    # per-protein identifier. diagnose_decomposition.py already writes
    # gain/loss/contacts; if it lacks a protein/position id, see note below.
    python case_study_lookup.py \\
        --csv diag_s2648/decomposition_s2648.csv \\
        --out case_studies_detail.csv

NOTE: if you want named proteins (e.g. T4 lysozyme), the decomposition CSV
needs a protein-id column. diagnose_decomposition.py currently records
wt/mut/contacts/components but may not carry the source protein name. If
absent, add it in collect_decomposition() (one line: store batch protein
id), re-run the diagnostic, then use this script. Without names you can
still show the burial→loss trend, just not label specific proteins.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# Large hydrophobic wildtype → smaller residue: the classic core-packing set
CORE_PACKING_WT = {"L", "I", "V", "F", "M", "W", "Y"}
SMALL_MUT = {"A", "G", "S"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="decomposition CSV with contacts column")
    p.add_argument("--out", default="case_studies_detail.csv")
    p.add_argument("--top", type=int, default=15, help="how many examples each end")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"Not found: {args.csv}")
    df = pd.read_csv(args.csv)

    required = {"wt", "mut", "gain_fwd", "loss_fwd", "ddg_true", "contacts"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}. Re-run diagnose_decomposition.py.")

    has_protein = any(c in df.columns for c in ("protein", "pdb", "name", "wt_name"))
    pcol = next((c for c in ("protein", "pdb", "name", "wt_name") if c in df.columns), None)

    print("=" * 72)
    print("CASE-STUDY LOOKUP (burial-conditional)")
    print("=" * 72)
    print(f"Loaded {len(df)} mutations from {args.csv}")
    print(f"Protein names available: {'YES (' + pcol + ')' if has_protein else 'NO'}")

    # ── Core-packing mutations, ranked by wild-type burial ────────
    core = df[df["wt"].isin(CORE_PACKING_WT) & df["mut"].isin(SMALL_MUT)].copy()
    core["abs_loss"] = core["loss_fwd"].abs()
    core["abs_gain"] = core["gain_fwd"].abs()
    print(f"\nCore-packing mutations (large hydrophobic → small): n={len(core)}")

    if len(core) >= 10:
        rl, _ = spearmanr(core["abs_loss"], core["contacts"])
        rg, _ = spearmanr(core["abs_gain"], core["contacts"])
        print(f"  Within this set:  ρ(|loss|,burial)={rl:.3f}   ρ(|gain|,burial)={rg:.3f}")
        print("  → Use THIS as the case-study claim: within real core-packing")
        print("    mutations, loss tracks burial more than gain does.")

    # Most-buried examples (the figure candidates)
    cols = [c for c in [pcol, "wt", "mut", "contacts", "gain_fwd", "loss_fwd",
                        "ddg_fwd", "ddg_true"] if c and c in core.columns]
    most_buried = core.nlargest(args.top, "contacts")[cols]
    least_buried = core.nsmallest(args.top, "contacts")[cols]

    print("\n--- MOST BURIED core-packing mutations (figure candidates) ---")
    print(most_buried.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n--- LEAST BURIED core-packing mutations (contrast) ---")
    print(least_buried.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Save full annotated set
    core_sorted = core.sort_values("contacts", ascending=False)[
        [c for c in cols] + ["abs_loss", "abs_gain"]
    ]
    core_sorted.to_csv(args.out, index=False, float_format="%.4f")
    print(f"\nSaved full core-packing set (burial-sorted) to: {args.out}")

    print("\n" + "=" * 72)
    print("HOW TO USE FOR THE FIGURE")
    print("=" * 72)
    print("""  1. If protein names are present, pick 3-5 well-characterized
     entries from the MOST BURIED list (e.g. a known cavity mutation).
  2. For each, the figure shows: contact number (burial), the gain and
     loss components, and predicted vs experimental ddG.
  3. The caption claim must be the burial-conditional one:
     'loss is amplified at buried positions', supported by the within-set
     correlation printed above — NOT a raw magnitude comparison.
  4. Every mechanism statement (e.g. 'L99A creates a hydrophobic cavity')
     must carry a literature citation; the decomposition values come from
     this CSV. Do not mix the two sources of authority.""")


if __name__ == "__main__":
    main()

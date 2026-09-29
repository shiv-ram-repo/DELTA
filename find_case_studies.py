"""
find_case_studies.py — Locate well-characterized mutations in DELTA output
===========================================================================

Cross-references a curated list of mechanistically well-understood protein
stability mutations against the decomposition CSV produced by
diagnose_decomposition.py. For each mutation found, reports DELTA's
gain/loss split and whether it matches the known mechanism.

This produces the raw material for the paper's case-study section — no new
computation, just spotlighting predictions you already made.

Usage:
    # First run diagnose_decomposition.py on each benchmark you want to search,
    # e.g. ssym_direct, s669, s2648 (these contain the classic model proteins).
    python diagnose_decomposition.py --data_root ... --ckpt ... --out diag_ssym --dataset ssym_direct
    python diagnose_decomposition.py --data_root ... --ckpt ... --out diag_s2648 --dataset s2648

    # Then search across the produced CSVs:
    python find_case_studies.py --csvs diag_ssym/decomposition_ssym_direct.csv \\
                                       diag_s2648/decomposition_s2648.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# Curated case-study mutations with KNOWN mechanisms
# ═══════════════════════════════════════════════════════════════════════
# Each entry: (wt, mut, mechanism_class, expected_pattern, citation_hint)
#
# mechanism_class:
#   "cavity"        — large→small hydrophobic in core; lose packing
#                     EXPECT: large loss, small gain
#   "overpack"      — small→large in core; steric strain
#                     EXPECT: large (unfavorable) gain
#   "charge_bury"   — introduce charge into hydrophobic core
#                     EXPECT: large unfavorable gain
#   "pro_helix"     — proline into helix/strand; backbone disruption
#                     EXPECT: large loss (wildtype backbone role)
#   "gly_flex"      — glycine introducing flexibility
#                     EXPECT: entropic, loss-dominated
#   "surface"       — surface substitution, near-neutral
#                     EXPECT: small gain AND small loss
#
# These are sequence-agnostic (wt,mut) pairs; the script reports ALL
# matching pairs found and you pick those at the famous positions.

CASE_STUDIES = [
    # T4 lysozyme cavity-creating mutations (Matthews lab, classic)
    ("L", "A", "cavity", "large_loss_small_gain", "T4 lysozyme L99A; ~160A^3 cavity, lose hydrophobic packing"),
    ("V", "A", "cavity", "large_loss_small_gain", "core Val->Ala cavity creation"),
    ("I", "A", "cavity", "large_loss_small_gain", "core Ile->Ala cavity creation"),
    ("F", "A", "cavity", "large_loss_small_gain", "aromatic->Ala, large cavity"),
    ("L", "G", "cavity", "large_loss_small_gain", "core Leu->Gly, cavity + flexibility"),

    # Overpacking / steric strain (small->large in core)
    ("A", "V", "overpack", "large_gain", "core Ala->Val overpacking"),
    ("A", "L", "overpack", "large_gain", "core Ala->Leu steric strain"),
    ("G", "A", "overpack", "moderate_gain", "Gly->Ala, loss of flexibility"),

    # Charge burial
    ("V", "D", "charge_bury", "large_gain", "buried hydrophobic->charged Asp"),
    ("L", "D", "charge_bury", "large_gain", "buried Leu->Asp, charge in core"),
    ("V", "E", "charge_bury", "large_gain", "buried Val->Glu"),
    ("I", "K", "charge_bury", "large_gain", "buried Ile->Lys"),

    # Proline disruption
    ("A", "P", "pro_helix", "large_loss", "Ala->Pro helix breaker"),
    ("L", "P", "pro_helix", "large_loss", "Leu->Pro backbone disruption"),
    ("E", "P", "pro_helix", "large_loss", "Glu->Pro helix disruption"),

    # Glycine flexibility
    ("A", "G", "gly_flex", "loss_dominated", "Ala->Gly added flexibility"),

    # Near-neutral surface (conservative)
    ("L", "I", "surface", "small_both", "conservative Leu<->Ile"),
    ("I", "V", "surface", "small_both", "conservative Ile->Val"),
    ("D", "E", "surface", "small_both", "conservative Asp->Glu"),
    ("S", "T", "surface", "small_both", "conservative Ser->Thr"),
    ("K", "R", "surface", "small_both", "conservative Lys->Arg"),
]


def check_pattern(gain, loss, expected):
    """Does the observed gain/loss match the expected mechanistic pattern?"""
    ag, al = abs(gain), abs(loss)
    if expected == "large_loss_small_gain":
        return al > ag and al > 0.3
    if expected == "large_gain":
        return ag > al and ag > 0.4
    if expected == "moderate_gain":
        return ag > al
    if expected == "large_loss":
        return al > ag and al > 0.3
    if expected == "loss_dominated":
        return al >= ag
    if expected == "small_both":
        return ag < 0.4 and al < 0.4
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csvs", nargs="+", required=True,
                   help="decomposition_*.csv files from diagnose_decomposition.py")
    p.add_argument("--out", default="case_studies.csv")
    p.add_argument("--min_count", type=int, default=1,
                   help="Minimum occurrences of a (wt,mut) pair to report")
    args = p.parse_args()

    # Load and concatenate all CSVs
    frames = []
    for pattern in args.csvs:
        for path in glob.glob(pattern):
            if os.path.exists(path):
                df = pd.read_csv(path)
                df["source"] = os.path.basename(path)
                frames.append(df)
                print(f"Loaded {len(df)} mutations from {path}")
    if not frames:
        print("No CSV files found.")
        return
    data = pd.concat(frames, ignore_index=True)
    print(f"\nTotal mutations across all files: {len(data)}\n")

    # Search for each case-study pattern
    results = []
    print("=" * 78)
    print("CASE-STUDY MUTATION SEARCH")
    print("=" * 78)

    for wt, mut, mech, expected, hint in CASE_STUDIES:
        matches = data[(data["wt"] == wt) & (data["mut"] == mut)]
        if len(matches) < args.min_count:
            continue

        mean_gain = matches["gain_fwd"].mean()
        mean_loss = matches["loss_fwd"].mean()
        mean_ddg_pred = matches["ddg_fwd"].mean()
        mean_ddg_true = matches["ddg_true"].mean()
        pattern_ok = check_pattern(mean_gain, mean_loss, expected)

        flag = "✓ MATCHES" if pattern_ok else "✗ differs"
        print(f"\n{wt}→{mut}  [{mech}]  (n={len(matches)})")
        print(f"  {hint}")
        print(f"  gain={mean_gain:+.3f}  loss={mean_loss:+.3f}  "
              f"ddg_pred={mean_ddg_pred:+.3f}  ddg_true={mean_ddg_true:+.3f}")
        print(f"  Expected: {expected}  →  {flag}")

        results.append({
            "wt": wt, "mut": mut, "mechanism": mech, "n": len(matches),
            "mean_gain": round(mean_gain, 3), "mean_loss": round(mean_loss, 3),
            "mean_ddg_pred": round(mean_ddg_pred, 3),
            "mean_ddg_true": round(mean_ddg_true, 3),
            "expected_pattern": expected,
            "pattern_matches": bool(pattern_ok),
            "mechanism_hint": hint,
        })

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.out, index=False)

        n_match = out_df["pattern_matches"].sum()
        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        print(f"  Case-study patterns found in data : {len(out_df)}")
        print(f"  Patterns matching known mechanism : {n_match}/{len(out_df)} "
              f"({100*n_match/len(out_df):.0f}%)")
        print(f"  Saved to: {args.out}")
        print("""
  For the paper: cite the ones that MATCH as evidence the decomposition
  recovers known biophysics. Be honest about the ones that don't — discuss
  them as cases where aggregate (wt,mut) averaging hides position-specific
  context (a surface L→A behaves differently from a core L→A).

  NOTE: these are AVERAGES over all instances of each (wt,mut) pair. For the
  strongest case study (e.g. the actual T4 lysozyme L99A), find that specific
  protein in your data and report the single-mutation gain/loss, not the
  average. Use the 'source' and per-row data in the decomposition CSVs.
""")
    else:
        print("\nNo case-study patterns found. Try running diagnose_decomposition.py")
        print("on more benchmarks (ssym_direct, s2648, s669 contain classic proteins).")


if __name__ == "__main__":
    main()

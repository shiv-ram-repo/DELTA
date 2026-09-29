"""
build_full_results_table.py — Complete Benchmark Table (R1, R3)
================================================================

Reviewers R1 and R3: Table 1 reported only 5 datasets and one metric, while
the paper describes 11 benchmarks and 4 metrics, and DELTA-B was referenced
but not shown. This script assembles the COMPLETE table from your eval CSVs
and emits it as LaTeX.

It reads the per-dataset metric CSVs produced by your evaluation
(protstab_train/evaluate.py -> test_metrics.csv with columns:
 dataset, n, spearman, pearson, r2, rmse, mae) for DELTA-U, DELTA-B, and any
seeds, and writes:
  full_results_table.tex   all benchmarks x {Spearman, Pearson, RMSE, MAE}
  full_results.csv         the merged machine-readable table

If multiple DELTA-U seed CSVs are given, per-cell mean+-std is reported for
Spearman (the CIs reviewers asked for).

Usage:
    python build_full_results_table.py \\
        --delta_u runs/delta_pretrained/eval/test_metrics.csv \\
        --delta_u_seeds runs/delta_s42/eval/test_metrics.csv runs/delta_s123/eval/test_metrics.csv \\
        --delta_b runs/delta_bidir/eval/test_metrics.csv \\
        --out results_out
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# Display order and pretty names
ORDER = ["megascale_test", "fireport_hf", "ssym_direct", "ssym_inverse",
         "s669", "s461", "s783", "s8754", "s2648", "s571", "s4346"]
PRETTY = {"megascale_test": "Megascale", "fireport_hf": "FireProt",
          "ssym_direct": "Ssym$_\\mathrm{dir}$", "ssym_inverse": "Ssym$_\\mathrm{inv}$",
          "s669": "S669", "s461": "S461", "s783": "S783", "s8754": "S8754",
          "s2648": "S2648", "s571": "S571", "s4346": "S4346"}

# S571 and S4346 measure ΔTm (melting-temperature shift, degrees), NOT ΔΔG
# (kcal/mol). Their RMSE/MAE are in different units and must not be compared
# against the ΔΔG error columns. Spearman/Pearson are scale-free and remain
# comparable. We flag these so the table reports their errors separately.
DTM_DATASETS = {"s571", "s4346"}


def load(path):
    if not path:
        return None
    if not os.path.exists(path):
        print(f"  WARNING: file not found, column will be empty: {path}")
        return None
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        print(f"  WARNING: {path} has no 'dataset' column (has: {list(df.columns)})")
        return None
    return df.set_index("dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta_u", default="")
    ap.add_argument("--delta_b", default="")
    ap.add_argument("--delta_u_seeds", nargs="*", default=[])
    ap.add_argument("--out", default="results_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    U = load(args.delta_u)
    B = load(args.delta_b)
    seed_frames = [load(p) for p in args.delta_u_seeds if load(p) is not None]

    if U is None and seed_frames:
        U = seed_frames[0]

    # seed mean/std for spearman
    seed_sp = None
    if seed_frames:
        seed_sp = pd.concat([f["spearman"] for f in seed_frames], axis=1)

    rows = []
    for ds in ORDER:
        row = {"dataset": PRETTY.get(ds, ds)}
        is_dtm = ds in DTM_DATASETS
        row["_is_dtm"] = is_dtm
        if U is not None and ds in U.index:
            row["U_sp"] = U.loc[ds, "spearman"]
            row["U_pe"] = U.loc[ds, "pearson"]
            # RMSE/MAE only comparable for ddG datasets; dTm errors are in
            # different units and reported as '--' in the ddG error columns.
            row["U_rmse"] = np.nan if is_dtm else U.loc[ds, "rmse"]
            row["U_mae"] = np.nan if is_dtm else U.loc[ds, "mae"]
        if seed_sp is not None and ds in seed_sp.index:
            row["U_sp_mean"] = seed_sp.loc[ds].mean()
            row["U_sp_std"] = seed_sp.loc[ds].std()
        if B is not None and ds in B.index:
            row["B_sp"] = B.loc[ds, "spearman"]
            row["B_pe"] = B.loc[ds, "pearson"]
            row["B_rmse"] = np.nan if is_dtm else B.loc[ds, "rmse"]
            row["B_mae"] = np.nan if is_dtm else B.loc[ds, "mae"]
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.out, "full_results.csv"), index=False)

    # ── LaTeX ─────────────────────────────────────────────────────
    def cell(v, fmt="{:.3f}"):
        return fmt.format(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else "--"

    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{Complete evaluation across all benchmarks. DELTA-U is the "
             r"primary shared-context model; DELTA-B the bidirectional ablation. "
             r"Where multiple seeds are available, Spearman is mean$\pm$std. "
             r"S571 and S4346 measure $\Delta T_m$ (thermal stability) rather than "
             r"$\Delta\Delta G$; their rank correlations are comparable but their "
             r"RMSE/MAE are in different units and are omitted ($-$) from the "
             r"$\Delta\Delta G$ error columns. SPURS values (published, matched "
             r"splits) are inserted by the authors for size-matched benchmarks; "
             r"S8754 and S4346 differ in retained mutation count and are so noted.}",
             r"\label{tab:full}",
             r"\begin{tabular}{l cccc cccc}",
             r"\toprule",
             r" & \multicolumn{4}{c}{DELTA-U} & \multicolumn{4}{c}{DELTA-B} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Benchmark & $\rho$ & $r$ & RMSE & MAE & $\rho$ & $r$ & RMSE & MAE \\",
             r"\midrule"]
    for r in tab.itertuples():
        d = r._asdict()
        usp = (f"{d['U_sp_mean']:.3f}$\\pm${d['U_sp_std']:.3f}"
               if d.get("U_sp_mean") == d.get("U_sp_mean") and "U_sp_mean" in d and not pd.isna(d.get("U_sp_mean"))
               else cell(d.get("U_sp")))
        line = (f"{d['dataset']} & {usp} & {cell(d.get('U_pe'))} & "
                f"{cell(d.get('U_rmse'))} & {cell(d.get('U_mae'))} & "
                f"{cell(d.get('B_sp'))} & {cell(d.get('B_pe'))} & "
                f"{cell(d.get('B_rmse'))} & {cell(d.get('B_mae'))} \\\\")
        lines.append(line)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    with open(os.path.join(args.out, "full_results_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote:")
    print(f"  {args.out}/full_results_table.tex   (all 11 benchmarks x 4 metrics)")
    print(f"  {args.out}/full_results.csv")
    print("\nPreview:")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

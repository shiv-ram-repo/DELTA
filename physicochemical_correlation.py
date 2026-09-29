"""
physicochemical_correlation.py — Orthogonal Physical Validation
================================================================

Addresses Reviewer 2's request to compare the learned gain/loss components
against INDEPENDENT physicochemical quantities (rather than only internal
consistency). No FoldX/Rosetta/MD needed: we use established residue-property
scales and the volume/hydrophobicity CHANGE of the substitution, which are
computed entirely from the mutation identity and require no extra tooling.

The mechanistic expectations we test (stated a priori, not fished):
  - loss magnitude should relate to how much is removed: the size and
    hydrophobicity of the WILD-TYPE residue (weighted by burial).
  - gain magnitude should relate to what is introduced: the properties of
    the MUTANT residue.
  - the net volume change and hydrophobicity change of the substitution
    should relate to the net predicted ddG (a basic sanity coupling).

We report Spearman correlations of |loss| and |gain| against:
  wt_volume, mut_volume, delta_volume, wt_hydropathy, mut_hydropathy,
  delta_hydropathy, and (burial-weighted) wt_volume*contacts.

This is orthogonal because none of these scales is used anywhere in training;
they are external residue properties.

Usage:
    python physicochemical_correlation.py \\
        --diag_csv diag_u_s2648/decomposition_s2648.csv \\
        --out physchem_out
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Kyte-Doolittle hydropathy (higher = more hydrophobic)
KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,
      "H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,
      "T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
# Zamyatnin residue van der Waals volumes (A^3)
VOL = {"A":88.6,"R":173.4,"N":114.1,"D":111.1,"C":108.5,"Q":143.8,"E":138.4,
       "G":60.1,"H":153.2,"I":166.7,"L":166.7,"K":168.6,"M":162.9,"F":189.9,
       "P":112.7,"S":89.0,"T":116.1,"W":227.8,"Y":193.6,"V":140.0}


def sr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan"), float("nan")
    r, p = spearmanr(a[m], b[m])
    return r, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag_csv", required=True)
    ap.add_argument("--out", default="physchem_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.diag_csv)
    for c in ("wt", "mut", "gain_fwd", "loss_fwd"):
        if c not in df.columns:
            raise SystemExit(f"CSV missing {c}")

    df["wt_vol"] = df["wt"].map(VOL)
    df["mut_vol"] = df["mut"].map(VOL)
    df["dvol"] = df["mut_vol"] - df["wt_vol"]
    df["wt_kd"] = df["wt"].map(KD)
    df["mut_kd"] = df["mut"].map(KD)
    df["dkd"] = df["mut_kd"] - df["wt_kd"]
    if "contacts" in df.columns:
        df["wt_vol_burial"] = df["wt_vol"] * df["contacts"]

    aloss = df["loss_fwd"].abs().to_numpy()
    again = df["gain_fwd"].abs().to_numpy()

    tests = [
        ("wt_volume",        "wt_vol"),
        ("mut_volume",       "mut_vol"),
        ("delta_volume",     "dvol"),
        ("wt_hydropathy",    "wt_kd"),
        ("mut_hydropathy",   "mut_kd"),
        ("delta_hydropathy", "dkd"),
    ]
    if "wt_vol_burial" in df.columns:
        tests.append(("wt_volume x burial", "wt_vol_burial"))

    rows = []
    print("=" * 72)
    print("PHYSICOCHEMICAL CORRELATION (orthogonal validation)")
    print("=" * 72)
    print(f"{'property':22s} {'rho(|loss|)':>12s} {'rho(|gain|)':>12s}")
    print("-" * 50)
    for name, col in tests:
        x = df[col].to_numpy()
        rl, pl = sr(aloss, x)
        rg, pg = sr(again, x)
        rows.append({"property": name, "rho_loss": round(rl, 3), "p_loss": pl,
                     "rho_gain": round(rg, 3), "p_gain": pg})
        print(f"{name:22s} {rl:>12.3f} {rg:>12.3f}")

    # Net sanity: does predicted ddg couple to volume/hydropathy change?
    if "ddg_fwd" in df.columns:
        dd = df["ddg_fwd"].to_numpy()
        r_dv, _ = sr(dd, df["dvol"].to_numpy())
        r_dk, _ = sr(dd, df["dkd"].to_numpy())
        print(f"\n  net ddg vs delta_volume:     rho={r_dv:.3f}")
        print(f"  net ddg vs delta_hydropathy: rho={r_dk:.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(args.out, "physchem_correlations.csv"), index=False)
    print(f"\nSaved: {args.out}/physchem_correlations.csv")
    print("""
  Expected pattern if components are mechanistically sensible:
    - |loss| couples more to WILD-TYPE properties (wt_volume, wt_hydropathy,
      and wt_volume x burial) than |gain| does;
    - |gain| couples more to MUTANT properties than |loss| does.
  Report the table and state which expectations hold and which do not.
  These scales are external to training, so agreement is orthogonal evidence;
  disagreement should be reported honestly as a limitation.
""")


if __name__ == "__main__":
    main()

"""
make_burial_merged.py — Single Merged Burial Figure
====================================================

Replaces the two separate burial figures (fig4 all-mutations and
fig_casestudy core-packing) with ONE two-panel figure:

  Left panel  : all mutations — |loss| and |gain| vs wild-type contacts
  Right panel : core-packing subset (large hydrophobic -> small)

This removes the redundancy of two same-axis figures while keeping both
scopes visible: the global modest association and its sharpening within the
mechanism-homogeneous subset.

Output: fig4_burial_merged.pdf / .png

Usage:
    python make_burial_merged.py \\
        --diag_u diag_u_s2648/decomposition_s2648.csv \\
        --out figures
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

LARGE_HYDRO = set("LIVFMWY")
SMALL = set("AGS")
C_LOSS, C_GAIN = "#2c6fbb", "#c0612a"

plt.rcParams.update({
    "font.size": 10, "font.family": "serif", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def panel(ax, d, title):
    d = d[d["contacts"] > 0]
    rl, _ = spearmanr(d["loss_fwd"].abs(), d["contacts"])
    rg, _ = spearmanr(d["gain_fwd"].abs(), d["contacts"])
    ax.scatter(d["contacts"], d["loss_fwd"].abs(), s=10, alpha=0.30, color=C_LOSS,
               label=fr"$|\ell|$ ($\rho$={rl:.2f})")
    ax.scatter(d["contacts"], d["gain_fwd"].abs(), s=10, alpha=0.30, color=C_GAIN,
               marker="^", label=fr"$|g|$ ($\rho$={rg:.2f})")
    bins = np.arange(0, d["contacts"].max() + 3, 3)
    db = d.assign(b=pd.cut(d["contacts"], bins))
    g = db.groupby("b", observed=True)
    ctr = [iv.mid for iv in g.groups.keys()]
    ax.plot(ctr, g["loss_fwd"].apply(lambda x: x.abs().mean()).values, "-",
            color=C_LOSS, lw=2)
    ax.plot(ctr, g["gain_fwd"].apply(lambda x: x.abs().mean()).values, "--",
            color=C_GAIN, lw=2)
    ax.set_xlabel(r"wild-type C$\alpha$ contacts ($<8$ \AA)")
    ax.set_title(title, fontsize=10)
    # Legend below the panel, outside the data area, to avoid collision with
    # the point cloud which is densest in the upper-left/upper-right.
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, handletextpad=0.4,
              columnspacing=1.2)
    # Headroom so the top points and binned-mean lines never touch the title.
    ymax = max(d["loss_fwd"].abs().max(), d["gain_fwd"].abs().max())
    ax.set_ylim(0, ymax * 1.08)
    return rl, rg, len(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diag_u", required=True)
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.diag_u)
    core = df[df["wt"].isin(LARGE_HYDRO) & df["mut"].isin(SMALL)]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=True)
    rl_a, rg_a, n_a = panel(axes[0], df, f"All mutations (n={len(df)})")
    rl_c, rg_c, n_c = panel(axes[1], core, f"Core-packing subset (n={len(core)})")
    axes[0].set_ylabel("component magnitude")
    fig.suptitle("Loss component tracks wild-type burial, sharply within core packing",
                 fontsize=10)

    base = os.path.join(args.out, "fig4_burial_merged")
    fig.subplots_adjust(bottom=0.26, wspace=0.08)
    fig.savefig(base + ".pdf"); fig.savefig(base + ".png", dpi=200)
    plt.close(fig)

    print(f"Wrote {base}.pdf / .png")
    print(f"  All mutations:  rho(|loss|)={rl_a:.3f}  rho(|gain|)={rg_a:.3f}  (n={n_a})")
    print(f"  Core-packing:   rho(|loss|)={rl_c:.3f}  rho(|gain|)={rg_c:.3f}  (n={n_c})")
    print("\nIn the paper, replace the two separate burial includegraphics with:")
    print(r"  \includegraphics[width=\linewidth]{figures/fig4_burial_merged.pdf}")


if __name__ == "__main__":
    main()

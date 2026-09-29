"""
make_figures.py — Publication Figures for the DELTA Paper
==========================================================

Generates all figures from the eval CSVs and decomposition CSVs you have
already produced. No retraining, no new computation beyond reading files.

Figures:
  Fig 1  (schematic)   — NOT generated here; draw in your vector tool.
  Fig 2  gain_loss_scatter      : gain vs loss colored by true ddG (DELTA-U)
  Fig 3  reverse_consistency    : gain_fwd vs loss_rev (the strongest diagnostic)
  Fig 4  burial_association     : |loss| and |gain| vs wildtype contact number
  Fig 5  mutation_type_heatmaps : 20x20 mean gain, loss, ddG per substitution
  Fig 6  u_vs_b_apportionment   : gain-fraction of destabilization, U vs B
  Fig 7  benchmark_bars         : Spearman per benchmark, DELTA-U vs DELTA-B (+SPURS if given)
  Fig 8  embedding_pca          : PCA of the learned mutation-type embedding

All figures are written as both PDF (vector, for the paper) and PNG (preview).
Styling is kept clean and journal-appropriate (no chartjunk, colorblind-safe).

Usage:
    python make_figures.py \\
        --diag_u diag_u_s2648/decomposition_s2648.csv \\
        --diag_b diag_bidir_s2648/decomposition_s2648.csv \\
        --eval_u runs/delta_pretrained/eval/test_metrics.csv \\
        --eval_b runs/delta_bidir/eval/test_metrics.csv \\
        --eval_spurs runs/spurs_baseline_eval/spurs_test_metrics.csv \\
        --embed_u runs/delta_pretrained/eval/learned_mut_type_embedding.npy \\
        --out figures/

Any argument may be omitted; the script generates whatever it can from
what it is given and reports what it skipped.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
CONSERVATIVE = {("L","I"),("I","V"),("V","I"),("D","E"),("E","D"),("K","R"),
                ("R","K"),("F","Y"),("Y","F"),("S","T"),("L","M"),("M","L"),
                ("I","L"),("V","L"),("L","V")}

# Clean journal style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def save(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.pdf"))
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ── Fig 2: gain vs loss scatter ───────────────────────────────────────
def fig_gain_loss_scatter(df, out):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sc = ax.scatter(df["loss_fwd"], df["gain_fwd"], c=df["ddg_true"],
                    cmap="coolwarm", s=8, alpha=0.6,
                    norm=TwoSlopeNorm(vcenter=0))
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    ax.axvline(0, color="0.6", lw=0.6, zorder=0)
    ax.set_xlabel(r"loss component  $\ell$")
    ax.set_ylabel(r"gain component  $g$")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"experimental $\Delta\Delta G$ (kcal/mol)")
    ax.set_title("Gain–loss decomposition (DELTA-U)")
    save(fig, out, "fig2_gain_loss_scatter")


# ── Fig 3: reverse consistency ────────────────────────────────────────
def fig_reverse_consistency(df, out):
    rho, _ = spearmanr(df["gain_fwd"], df["loss_rev"])
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.scatter(df["gain_fwd"], df["loss_rev"], s=8, alpha=0.4, color="#2b6cb0")
    lims = [min(df["gain_fwd"].min(), df["loss_rev"].min()),
            max(df["gain_fwd"].max(), df["loss_rev"].max())]
    ax.plot(lims, lims, "--", color="0.5", lw=0.8)
    ax.set_xlabel(r"forward gain of residue $b$,  $g^{\mathrm{fwd}}(b)$")
    ax.set_ylabel(r"reverse loss of residue $b$,  $\ell^{\mathrm{rev}}(b)$")
    ax.set_title(f"Reverse-consistency  (Spearman $\\rho$ = {rho:.2f})")
    save(fig, out, "fig3_reverse_consistency")


# ── Fig 4: burial association ─────────────────────────────────────────
def fig_burial(df, out):
    if "contacts" not in df.columns:
        print("  skip fig4: no contacts column"); return
    v = df[df["contacts"] > 0].copy()
    # bin by contact number for a clean trend
    bins = np.linspace(v["contacts"].min(), v["contacts"].max(), 12)
    v["bin"] = pd.cut(v["contacts"], bins)
    grp = v.groupby("bin", observed=True).agg(
        c=("contacts", "mean"),
        loss=("loss_fwd", lambda x: x.abs().mean()),
        gain=("gain_fwd", lambda x: x.abs().mean()),
    ).dropna()
    rl, _ = spearmanr(v["loss_fwd"].abs(), v["contacts"])
    rg, _ = spearmanr(v["gain_fwd"].abs(), v["contacts"])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot(grp["c"], grp["loss"], "o-", color="#c0392b", ms=4,
            label=fr"$|\ell|$  ($\rho$={rl:.2f})")
    ax.plot(grp["c"], grp["gain"], "s-", color="#2b6cb0", ms=4,
            label=fr"$|g|$  ($\rho$={rg:.2f})")
    ax.set_xlabel(r"wild-type C$\alpha$ contact number (burial)")
    ax.set_ylabel("mean component magnitude")
    ax.legend(frameon=False)
    ax.set_title("Loss tracks burial more than gain")
    save(fig, out, "fig4_burial_association")


# ── Fig 5: mutation-type heatmaps ─────────────────────────────────────
def fig_heatmaps(df, out):
    gmap = np.full((20, 20), np.nan)
    lmap = np.full((20, 20), np.nan)
    dmap = np.full((20, 20), np.nan)
    for wi, wt in enumerate(ALPHABET):
        for mi, mut in enumerate(ALPHABET):
            sel = df[(df["wt"] == wt) & (df["mut"] == mut)]
            if len(sel):
                gmap[wi, mi] = sel["gain_fwd"].mean()
                lmap[wi, mi] = sel["loss_fwd"].mean()
                dmap[wi, mi] = sel["ddg_fwd"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    for ax, M, title in zip(axes, [gmap, lmap, dmap],
                            ["mean gain $g$", "mean loss $\\ell$", "mean $\\Delta\\Delta G$"]):
        im = ax.imshow(M, cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0), aspect="equal")
        ax.set_xticks(range(20)); ax.set_xticklabels(ALPHABET, fontsize=5.5)
        ax.set_yticks(range(20)); ax.set_yticklabels(ALPHABET, fontsize=5.5)
        ax.set_xlabel("mutant"); ax.set_ylabel("wild-type")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Substitution-type decomposition (rows: wild-type, cols: mutant)", y=1.02)
    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.35, top=0.88, bottom=0.12)
    save(fig, out, "fig5_mutation_type_heatmaps")


# ── Fig 6: U vs B apportionment ───────────────────────────────────────
def fig_apportionment(df_u, df_b, out):
    def frac(df):
        d = df[df["ddg_true"] > 1.0]
        g = d["gain_fwd"].mean(); l = -d["loss_fwd"].mean()
        return g / (g + l + 1e-9)
    fu = frac(df_u) if df_u is not None else None
    fb = frac(df_b) if df_b is not None else None
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    labels, gains = [], []
    if fu is not None: labels.append("DELTA-U"); gains.append(fu)
    if fb is not None: labels.append("DELTA-B"); gains.append(fb)
    x = np.arange(len(labels))
    ax.bar(x, gains, color="#2b6cb0", width=0.5, label="gain")
    ax.bar(x, [1-g for g in gains], bottom=gains, color="#c0392b", width=0.5, label="loss")
    ax.axhline(0.5, color="0.4", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of destabilization")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Branch apportionment")
    save(fig, out, "fig6_apportionment")


# ── Fig 7: benchmark bars ─────────────────────────────────────────────
def fig_benchmarks(eval_u, eval_b, eval_spurs, out):
    series = {}
    for label, path in [("DELTA-U", eval_u), ("DELTA-B", eval_b), ("SPURS", eval_spurs)]:
        if path and os.path.exists(path):
            series[label] = pd.read_csv(path).set_index("dataset")["spearman"]
    if not series:
        print("  skip fig7: no eval CSVs"); return
    table = pd.DataFrame(series)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(table.index))
    w = 0.8 / len(table.columns)
    colors = {"DELTA-U": "#2b6cb0", "DELTA-B": "#27ae60", "SPURS": "#7f8c8d"}
    for i, col in enumerate(table.columns):
        ax.bar(x + i*w, table[col].values, w, label=col,
               color=colors.get(col, None))
    ax.set_xticks(x + w*(len(table.columns)-1)/2)
    ax.set_xticklabels(table.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.legend(frameon=False)
    ax.set_title("Per-benchmark accuracy")
    save(fig, out, "fig7_benchmark_bars")


# ── Fig 8: embedding PCA ──────────────────────────────────────────────
def fig_embedding_pca(embed_path, out):
    if not (embed_path and os.path.exists(embed_path)):
        print("  skip fig8: no embedding .npy"); return
    E = np.load(embed_path)  # (20, 20, d)
    # Represent each substitution (wt,mut) by its embedding vector
    vecs, labels, colors = [], [], []
    # Color by hydrophobicity change as one interpretable axis
    kd = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,
          "H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,
          "T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
    for wi, wt in enumerate(ALPHABET):
        for mi, mut in enumerate(ALPHABET):
            if wt == mut: continue
            vecs.append(E[wi, mi]); labels.append(f"{wt}{mut}")
            colors.append(kd[mut] - kd[wt])  # hydrophobicity change
    vecs = np.array(vecs)
    vecs = vecs - vecs.mean(0)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(vecs, full_matrices=False)
    pc = U[:, :2] * S[:2]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=colors, cmap="PuOr", s=10, alpha=0.7)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Δ hydropathy (mut − wt)")
    ax.set_title("Learned mutation-type embedding (PCA)")
    save(fig, out, "fig8_embedding_pca")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diag_u", default="")
    p.add_argument("--diag_b", default="")
    p.add_argument("--eval_u", default="")
    p.add_argument("--eval_b", default="")
    p.add_argument("--eval_spurs", default="")
    p.add_argument("--embed_u", default="")
    p.add_argument("--out", default="figures")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df_u = pd.read_csv(args.diag_u) if args.diag_u and os.path.exists(args.diag_u) else None
    df_b = pd.read_csv(args.diag_b) if args.diag_b and os.path.exists(args.diag_b) else None

    print("Generating figures...")
    if df_u is not None:
        fig_gain_loss_scatter(df_u, args.out)
        fig_reverse_consistency(df_u, args.out)
        fig_burial(df_u, args.out)
        fig_heatmaps(df_u, args.out)
        fig_embedding_pca(args.embed_u, args.out)
    else:
        print("  skip figs 2-5,8: no DELTA-U decomposition CSV")
    if df_u is not None or df_b is not None:
        fig_apportionment(df_u, df_b, args.out)
    fig_benchmarks(args.eval_u, args.eval_b, args.eval_spurs, args.out)

    print(f"\nDone. Figures in {args.out}/ (PDF for paper, PNG for preview).")
    print("Fig 1 (architecture schematic) is not auto-generated; draw it in a")
    print("vector tool using the dimensions from the Methods section.")


if __name__ == "__main__":
    main()

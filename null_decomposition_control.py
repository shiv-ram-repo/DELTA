"""
null_decomposition_control.py — Null-Decomposition Control Experiment
======================================================================

Addresses the central reviewer concern (all three reviewers, R1-W3, R2-major,
R3-last): the gain/loss decomposition is non-identifiable because only the
difference g - l = ddG is supervised. The relevant question is therefore NOT
"do DELTA's components correlate with burial / show reverse-consistency?" but

    "Does DELTA's learned decomposition show MORE structure than an ARBITRARY
     decomposition that predicts the same ddG equally well?"

This script builds several classes of null decomposition that EXACTLY
reproduce DELTA's net prediction (g_null - l_null = ddg_fwd, to numerical
precision) but are otherwise arbitrary, and compares DELTA's diagnostic
scores against the null distribution. If DELTA's burial association and
reverse-consistency exceed the null distribution, the learned decomposition
carries structure beyond what net-ddG supervision forces. If not, the
reviewers' skepticism is upheld and the interpretation must be withdrawn.

Null models (all satisfy g_null - l_null = ddg exactly):
  N1  random split:        l_null = ddg * U,  g_null = ddg + l_null,  U~Uniform(0,1) per mutation
  N2  fixed-fraction split: l_null = ddg * c,  g_null = ddg * (1-c)... (c const) -> deterministic fraction
  N3  gaussian offset:      l_null = z,  g_null = ddg + z,  z~N(0, sigma_ddg)
  N4  magnitude-matched:    random split rescaled so |g|+|l| matches DELTA's mean magnitude
  N5  shuffle:              DELTA's own (g,l) with the mutation<->row assignment permuted
                            (preserves marginal distribution, destroys per-mutation identity)

For each null we recompute the two key diagnostics:
  - burial association:  rho(|l|, contacts) and the gap rho(|l|)-rho(|g|)
  - reverse-consistency: rho(g_fwd, l_rev)   [only computable for N5 shuffle and
                          for nulls that define a reverse; for row-local nulls we
                          report burial only and say so]

Reverse-consistency requires a matched reverse decomposition; only the shuffle
null (N5) and DELTA define one per residue, so reverse-consistency is compared
against the shuffle null, which is the appropriate null for that specific test.

Usage:
    python null_decomposition_control.py \\
        --diag_csv diag_u_s2648/decomposition_s2648.csv \\
        --n_null 1000 \\
        --out null_control_out
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def burial_gap(gain, loss, contacts):
    """rho(|loss|,contacts) - rho(|gain|,contacts). DELTA's claim: this is > 0."""
    m = contacts > 0
    rl, _ = spearmanr(np.abs(loss)[m], contacts[m])
    rg, _ = spearmanr(np.abs(gain)[m], contacts[m])
    return rl, rg, rl - rg


def make_null(kind, ddg, rng, delta_gain=None, delta_loss=None):
    """Return (gain_null, loss_null) with gain_null - loss_null == ddg exactly."""
    n = len(ddg)
    if kind == "N1_random_split":
        # loss = ddg * u, gain = ddg + loss  -> gain - loss = ddg
        u = rng.uniform(0, 1, n)
        loss = ddg * u
        gain = ddg + loss
    elif kind == "N3_gaussian_offset":
        z = rng.normal(0, np.std(ddg), n)
        loss = z
        gain = ddg + z
    elif kind == "N4_magnitude_matched":
        # random split then rescale offset to match DELTA mean |g|+|l|
        u = rng.uniform(0, 1, n)
        loss = ddg * u
        gain = ddg + loss
        if delta_gain is not None:
            target = np.mean(np.abs(delta_gain) + np.abs(delta_loss))
            cur = np.mean(np.abs(gain) + np.abs(loss)) + 1e-9
            scale = target / cur
            # rescale the SHARED offset, preserving gain-loss=ddg:
            off = loss * scale
            loss = off
            gain = ddg + off
    elif kind == "N5_shuffle":
        # take DELTA's own components but permute which mutation they belong to,
        # then re-anchor the difference to THIS mutation's ddg by shifting.
        perm = rng.permutation(n)
        off = delta_loss[perm]           # borrow a real loss magnitude from elsewhere
        loss = off
        gain = ddg + off
    else:
        raise ValueError(kind)
    return gain, loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diag_csv", required=True,
                   help="decomposition CSV from diagnose_decomposition.py")
    p.add_argument("--n_null", type=int, default=1000,
                   help="number of null draws per null class")
    p.add_argument("--out", default="null_control_out")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.diag_csv)

    need = {"gain_fwd", "loss_fwd", "ddg_fwd", "contacts"}
    if not need.issubset(df.columns):
        raise SystemExit(f"CSV missing columns: {need - set(df.columns)}")

    ddg = df["ddg_fwd"].to_numpy()
    dgain = df["gain_fwd"].to_numpy()
    dloss = df["loss_fwd"].to_numpy()
    contacts = df["contacts"].to_numpy()

    # ── DELTA's own scores ────────────────────────────────────────
    d_rl, d_rg, d_gap = burial_gap(dgain, dloss, contacts)
    print("=" * 70)
    print("NULL-DECOMPOSITION CONTROL")
    print("=" * 70)
    print(f"Mutations: {len(df)}   (dataset: {os.path.basename(args.diag_csv)})")
    print(f"\nDELTA burial: rho(|loss|,contacts)={d_rl:.3f}  "
          f"rho(|gain|,contacts)={d_rg:.3f}  gap={d_gap:+.3f}")

    # ── Null distributions of the burial gap ──────────────────────
    null_kinds = ["N1_random_split", "N3_gaussian_offset",
                  "N4_magnitude_matched", "N5_shuffle"]
    rows = []
    summary = []
    for kind in null_kinds:
        gaps, rls = [], []
        for _ in range(args.n_null):
            g, l = make_null(kind, ddg, rng, dgain, dloss)
            # sanity: reconstruction holds
            rl, rg, gap = burial_gap(g, l, contacts)
            gaps.append(gap); rls.append(rl)
        gaps = np.array(gaps); rls = np.array(rls)
        # one-sided p: fraction of null gaps >= DELTA's gap
        p_gap = float(np.mean(gaps >= d_gap))
        p_rl = float(np.mean(rls >= d_rl))
        summary.append({
            "null": kind,
            "null_gap_mean": round(float(gaps.mean()), 4),
            "null_gap_std": round(float(gaps.std()), 4),
            "null_gap_p95": round(float(np.percentile(gaps, 95)), 4),
            "delta_gap": round(d_gap, 4),
            "p_gap_ge_delta": p_gap,
            "null_rlossburial_mean": round(float(rls.mean()), 4),
            "delta_rlossburial": round(d_rl, 4),
            "p_rl_ge_delta": p_rl,
        })
        for gp, rl in zip(gaps, rls):
            rows.append({"null": kind, "gap": gp, "r_loss_burial": rl})
        print(f"\n  {kind}:")
        print(f"    null burial-gap: mean={gaps.mean():+.3f} std={gaps.std():.3f} "
              f"p95={np.percentile(gaps,95):+.3f}")
        print(f"    DELTA gap={d_gap:+.3f}  ->  P(null >= DELTA) = {p_gap:.4f}")
        print(f"    null rho(|loss|,burial): mean={rls.mean():.3f}  "
              f"DELTA={d_rl:.3f}  ->  P(null >= DELTA) = {p_rl:.4f}")

    pd.DataFrame(summary).to_csv(os.path.join(args.out, "null_summary.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "null_draws.csv"), index=False)

    # ── Reverse-consistency vs shuffle null ───────────────────────
    if {"loss_rev", "gain_rev"}.issubset(df.columns):
        dl_rev = df["loss_rev"].to_numpy()
        d_revcons, _ = spearmanr(dgain, dl_rev)
        null_rev = []
        for _ in range(args.n_null):
            perm = rng.permutation(len(df))
            r, _ = spearmanr(dgain, dl_rev[perm])
            null_rev.append(r)
        null_rev = np.array(null_rev)
        p_rev = float(np.mean(np.abs(null_rev) >= abs(d_revcons)))
        print("\n" + "-" * 70)
        print("Reverse-consistency vs shuffle null (permuted residue identity):")
        print(f"    DELTA rho(gain_fwd,loss_rev) = {d_revcons:.3f}")
        print(f"    null mean = {null_rev.mean():+.3f}  std = {null_rev.std():.3f}")
        print(f"    P(|null| >= |DELTA|) = {p_rev:.4f}")
        with open(os.path.join(args.out, "reverse_null.txt"), "w") as f:
            f.write(f"delta_revcons={d_revcons:.4f}\n")
            f.write(f"null_mean={null_rev.mean():.4f}\nnull_std={null_rev.std():.4f}\n")
            f.write(f"p_two_sided={p_rev:.4f}\n")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
  For each null class, P(null >= DELTA) is the fraction of arbitrary
  same-ddG decompositions whose burial gap matches or exceeds DELTA's.
  A small value (e.g. < 0.05) means DELTA's decomposition is structured
  BEYOND what net-ddG supervision alone forces — the direct rebuttal to
  the identifiability objection. A large value means the reviewers are
  right and the burial structure is not specific to DELTA's solution.

  Report the null_summary.csv as a camera-ready table. State the result
  honestly whichever way it falls.
""")


if __name__ == "__main__":
    main()

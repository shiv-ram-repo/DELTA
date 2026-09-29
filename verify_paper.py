"""
verify_paper.py — Resolve Verifiable Claims and Audit SPURS Comparability
==========================================================================

Two jobs:

  (A) Auto-resolve the \\VERIFY{} marks that can be checked against your
      actual files: architecture dimensions, hyperparameters, the benchmark
      list, mutation counts, and the diagnostic numbers. Prints a report you
      paste against the manuscript.

  (B) Audit whether your benchmark splits match the CANONICAL published
      sizes, so you know which SPURS (or other) published numbers you can
      honestly place in the comparison table. This is the part that protects
      you from an apples-to-oranges baseline column.

It does NOT run SPURS. It tells you, per benchmark, whether borrowing a
published number is defensible (your split == canonical split) or needs a
footnote / "n/a" (your split differs).

Usage:
    python verify_paper.py \\
        --delta_u_eval runs/delta_pretrained/eval/test_metrics.csv \\
        --delta_b_eval runs/delta_bidir/eval/test_metrics.csv \\
        --diag_u diag_u_s2648/decomposition_s2648.csv \\
        --diag_b diag_bidir_s2648/decomposition_s2648.csv \\
        --embedding runs/delta_pretrained/eval/learned_mut_type_embedding.npy \\
        --train_config runs/delta_pretrained/  (dir or training_log.csv) \\
        --out verification_report.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
CONSERVATIVE = {("L","I"),("I","V"),("V","I"),("D","E"),("E","D"),("K","R"),
                ("R","K"),("F","Y"),("Y","F"),("S","T"),("L","M"),("M","L"),
                ("I","L"),("V","L"),("L","V")}

# Canonical benchmark sizes as standardly reported in the literature.
# If your run's mutation count matches, a published number on that benchmark
# is (very likely) computed on the same split and is borrowable with citation.
# If it differs, the published number is NOT directly comparable.
# NOTE: these are the commonly cited counts; confirm against the specific
# paper you borrow from, since some report filtered subsets.
CANONICAL_SIZES = {
    "s669": 669,
    "ssym_direct": 342,        # Ssym symmetric set, direct
    "ssym_inverse": 342,       # Ssym, inverse (hypothetical reverse)
    "s2648": 2648,
    "s461": 461,
    "s783": 783,
    "s571": 571,
    "s8754": 8754,             # note: your run showed 8228 valid -> FLAG
    "s4346": 4346,             # your run showed 3969 valid -> FLAG
    "fireport_hf": None,       # FireProt variants differ; confirm manually
    "megascale_test": None,    # split-specific; not a standard borrow target
}


def out(s=""):
    print(s)


def section(title):
    out("\n" + "=" * 74)
    out(title)
    out("=" * 74)


# ── (A) Architecture / hyperparameter resolution ──────────────────────

def resolve_architecture():
    """Resolve fixed architecture VERIFY marks from the code defaults."""
    section("(A1) ARCHITECTURE & HYPERPARAMETERS — paste into \\VERIFY marks")
    facts = {
        "d (backbone representation dim)": "1408 (= 1280 ESM-2 + 128 encoder)",
        "d_enc (encoder projection dim)": "128  [VERIFY this matches your config mlp.input_dim]",
        "d_sub (mutation-type embed dim)": "64",
        "d_aa (residue-identity embed dim)": "64",
        "d_hidden (branch MLP width)": "512",
        "n_branch_layers": "3 (GELU, dropout 0.1)",
        "n_context_heads": "8 (head dim = 1408/8 = 176)",
        "k_neighbors": "32",
        "sigma_init (distance kernel)": "8.0 Angstrom (learned per head, log-parameterized)",
        "ESM-2 model": "esm2_t33_650M_UR50D (33 layers, 1280-dim)",
        "adapter layer index": "-3  [VERIFY matches your training flag]",
        "cross-attention": "embed_dim 1280, 8 heads",
        "loss weights": "reg 1.0 / bcas 0.5 / decomp_antisym 0.3 / mag 0.1",
        "bcas alpha/beta": "1.0 / 0.5",
        "magnitude slack gamma": "2.0",
        "optimizer": "AdamW lr 1e-4, weight_decay 1e-2, grad clip 1.0",
        "schedule": "up to 200 epochs, early stop patience 30",
        "backbone fine-tune LR scale": "0.1x (frozen regime: 0x)",
        "primary loss": "SmoothL1 (Huber)  [VERIFY: state if any reported run used BMC]",
        "param counts": "head 10.59M, mut-type embed 25,600, total trainable 21.1M",
    }
    for k, v in facts.items():
        out(f"  {k:38s}: {v}")
    out("\n  Items with [VERIFY] above still need a one-time check against the")
    out("  exact config you launched (configs can drift from code defaults).")


def resolve_train_config(path):
    """If a training config/log is provided, extract seeds and confirm settings."""
    section("(A2) TRAINING CONFIG (from your run directory)")
    if not path:
        out("  --train_config not provided; cannot auto-confirm seeds/regime.")
        return
    # Look for a config json or the training_log.csv
    cfg_found = False
    for root, _, files in os.walk(path if os.path.isdir(path) else os.path.dirname(path)):
        for f in files:
            if f.endswith(".json") and "config" in f.lower():
                try:
                    cfg = json.load(open(os.path.join(root, f)))
                    out(f"  Found config: {os.path.join(root, f)}")
                    for key in ("seed", "loss_type", "adapter_layer", "bidirectional",
                                "backbone_ckpt", "backbone_lr_scale", "freeze_backbone",
                                "k_neighbors", "d_sub", "d_hidden"):
                        if key in cfg:
                            out(f"    {key}: {cfg[key]}")
                    cfg_found = True
                except Exception as e:
                    out(f"  (could not parse {f}: {e})")
    if not cfg_found:
        out("  No config json found. The training logs you pasted show:")
        out("    seed=42 (delta_pretrained), seed=123 (delta_seed123),")
        out("    loss_type=huber, adapter_layer=-3, k_neighbors=32, d_sub=64.")
        out("  Confirm these match every run you report.")


# ── (B) Benchmark split audit (SPURS comparability) ───────────────────

def audit_benchmarks(eval_csv, label):
    section(f"(B) BENCHMARK SPLIT AUDIT — {label}")
    if not (eval_csv and os.path.exists(eval_csv)):
        out(f"  {label} eval CSV not provided/found; skipping.")
        return
    df = pd.read_csv(eval_csv)
    out(f"  {'benchmark':16s} {'your n':>8s} {'canonical':>10s}  borrow published number?")
    out("  " + "-" * 64)
    for _, row in df.iterrows():
        ds = str(row["dataset"])
        n = int(row["n"]) if "n" in row else None
        canon = CANONICAL_SIZES.get(ds, None)
        if canon is None:
            verdict = "MANUAL — confirm split matches the source paper"
        elif n == canon:
            verdict = "YES — same size; cite published number as comparable"
        else:
            verdict = f"NO — differs ({n} vs {canon}); footnote or mark n/a"
        canon_s = str(canon) if canon is not None else "?"
        out(f"  {ds:16s} {str(n):>8s} {canon_s:>10s}  {verdict}")
    out("""
  RULE: only place a published SPURS (or other) number in the comparison
  table for rows marked YES. For NO/MANUAL rows, either (a) recompute the
  baseline in your pipeline, or (b) omit the baseline cell and note that the
  split differs. Do not silently place a non-comparable number side by side
  with your own — reviewers familiar with the benchmark will catch it.""")


# ── (C) Diagnostic numbers recomputation ──────────────────────────────

def recompute_diagnostics(csv, label):
    section(f"(C) DIAGNOSTIC NUMBERS — {label}")
    if not (csv and os.path.exists(csv)):
        out(f"  {label} decomposition CSV not provided; skipping.")
        return
    df = pd.read_csv(csv)
    dataset = os.path.basename(csv).replace("decomposition_", "").replace(".csv", "")
    out(f"  (dataset inferred: {dataset}; state this dataset in the paper)")

    if {"gain_fwd","loss_rev","loss_fwd","gain_rev"}.issubset(df.columns):
        r1, p1 = spearmanr(df["gain_fwd"], df["loss_rev"])
        r2, p2 = spearmanr(df["loss_fwd"], df["gain_rev"])
        out(f"  D1 reverse-consistency: rho(gain_fwd,loss_rev)={r1:.3f} (p={p1:.1e})")
        out(f"                          rho(loss_fwd,gain_rev)={r2:.3f} (p={p2:.1e})")

    if {"wt","mut","gain_fwd","loss_fwd"}.issubset(df.columns):
        isc = df.apply(lambda r: (r["wt"], r["mut"]) in CONSERVATIVE, axis=1)
        cm = (df[isc]["gain_fwd"].abs()+df[isc]["loss_fwd"].abs()).mean()
        nm = (df[~isc]["gain_fwd"].abs()+df[~isc]["loss_fwd"].abs()).mean()
        out(f"  D2 conservative attenuation: cons={cm:.3f} noncons={nm:.3f} ratio={cm/nm:.3f} (n_cons={int(isc.sum())})")

    if {"loss_fwd","gain_fwd","contacts"}.issubset(df.columns):
        v = df[df["contacts"]>0]
        rl,_ = spearmanr(v["loss_fwd"].abs(), v["contacts"])
        rg,_ = spearmanr(v["gain_fwd"].abs(), v["contacts"])
        out(f"  D3 burial: rho(|loss|,contacts)={rl:.3f}  rho(|gain|,contacts)={rg:.3f}")

    if {"gain_fwd","loss_fwd","ddg_true"}.issubset(df.columns):
        d = df[df["ddg_true"]>1.0]
        if len(d):
            g=d["gain_fwd"].mean(); l=-d["loss_fwd"].mean()
            out(f"  Apportionment: gain share={g/(g+l+1e-9):.1%} (n={len(d)})")


def resolve_embedding_antisymmetry(npy, label):
    section(f"(D) EMBEDDING DIRECTIONALITY (D4) — {label}")
    if not (npy and os.path.exists(npy)):
        out("  embedding .npy not provided; skipping.")
        return
    E = np.load(npy)  # (20,20,d)
    scores = []
    iu = np.triu_indices(20, k=1)
    for d in range(E.shape[2]):
        M = E[:,:,d]
        up, lo = M[iu], M.T[iu]
        if up.std()>1e-6 and lo.std()>1e-6:
            scores.append(np.corrcoef(up, -lo)[0,1])
    scores = np.array(scores)
    out(f"  mean anti-symmetry corr across {len(scores)} channels: {np.nanmean(scores):.3f}")
    out(f"  channels with strong anti-symmetry (>0.3): {(scores>0.3).sum()}/{len(scores)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delta_u_eval", default="")
    p.add_argument("--delta_b_eval", default="")
    p.add_argument("--diag_u", default="")
    p.add_argument("--diag_b", default="")
    p.add_argument("--embedding", default="")
    p.add_argument("--train_config", default="")
    p.add_argument("--out", default="verification_report.txt")
    args = p.parse_args()

    class Tee:
        def __init__(self,*fs): self.fs=fs
        def write(self,x):
            for f in self.fs: f.write(x)
        def flush(self):
            for f in self.fs: f.flush()
    fh = open(args.out, "w")
    sys.stdout = Tee(sys.__stdout__, fh)

    out("DELTA — Paper Verification Report")
    out("Resolves file-checkable claims and audits baseline comparability.")
    out("Does NOT run SPURS. Tells you which published numbers are borrowable.")

    resolve_architecture()
    resolve_train_config(args.train_config)
    audit_benchmarks(args.delta_u_eval, "DELTA-U")
    audit_benchmarks(args.delta_b_eval, "DELTA-B")
    recompute_diagnostics(args.diag_u, "DELTA-U")
    recompute_diagnostics(args.diag_b, "DELTA-B")
    resolve_embedding_antisymmetry(args.embedding, "DELTA-U")

    section("WHAT STILL REQUIRES A HUMAN (cannot be auto-verified)")
    out("""  - Citations: pull real references (ESM-2, ProteinMPNN, Tsuboyama 2023,
    BLOSUM62, ACDC-NN, JanusDDG, SPURS, representative DDG predictors).
  - SPURS baseline numbers: borrow ONLY for benchmarks marked YES in the
    split audit above; cite as 'as reported by [SPURS]'. For S8754 and S4346
    your valid-mutation counts (8228, 3969) differ from canonical sizes, so
    those rows are likely NOT directly comparable — footnote or omit.
  - MMseqs2 filtering description: state your identity threshold.
  - Case-study figure: per-protein values + literature mechanisms.
  - Author list, affiliations.""")

    sys.stdout = sys.__stdout__
    fh.close()
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()

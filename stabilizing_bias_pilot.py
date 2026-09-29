"""
stabilizing_bias_pilot.py — Is the Destabilization Bias a Gain/Loss Asymmetry?
==============================================================================

MOTIVATION (from literature): SOTA DDG models are biased toward predicting
destabilization; e.g. ThermoMPNN has ~46% PPV for stabilizing mutations on
Megascale. The field treats this as a data-imbalance problem. DELTA's gain/
loss decomposition lets us test a MECHANISTIC hypothesis nobody has checked:

    H: the bias arises because the GAIN component is under-calibrated
       relative to the LOSS component. A stabilizing mutation needs
       gain > loss; if gain is systematically compressed (because
       stabilizing / large-gain examples are rare in training), the model
       cannot express stabilization, producing the destabilization bias.

This pilot decides whether that hypothesis is worth building an algorithm
around. It requires NO training — only the decomposition CSVs already
produced by diagnose_decomposition.py (with gain_fwd, loss_fwd, ddg_fwd,
ddg_true).

It answers four questions:

 Q1  Is the model actually destabilization-biased on THIS data?
     (mean predicted ddg vs mean true ddg; fraction predicted stabilizing
      vs actually stabilizing; stabilizing-mutation PPV.)

 Q2  Do gain and loss occupy different effective scales / ranges?
     (std, IQR, dynamic range of gain vs loss; are large-gain events rarer
      than large-loss events?)

 Q3  Is the error on STABILIZING mutations specifically attributable to the
     gain side? Split mutations into stabilizing (ddg_true<0) vs
     destabilizing; for each, is prediction error correlated with gain
     magnitude more than loss magnitude?

 Q4  Counterfactual scale-match test: if we rescale the gain component so
     its spread matches the loss component's, does stabilizing-mutation
     recovery improve WITHOUT hurting overall correlation? (A cheap proxy
     for whether a scale-coupling algorithm could work.)

Usage:
    python stabilizing_bias_pilot.py \\
        --csvs diag_seed42/decomposition_s2648.csv \\
               diag_seed123/decomposition_s2648.csv \\
        --stab_threshold -0.5 \\
        --out stab_pilot_out
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_all(paths):
    frames = []
    for p in paths:
        if os.path.exists(p):
            d = pd.read_csv(p)
            d["_src"] = os.path.basename(os.path.dirname(p)) or os.path.basename(p)
            frames.append(d)
            print(f"  loaded {len(d)} from {p}")
        else:
            print(f"  MISSING {p}")
    if not frames:
        raise SystemExit("no CSVs found")
    return frames


def q1_bias(df, thr):
    print("\n" + "=" * 70)
    print("Q1 — Is the model destabilization-biased here?")
    print("=" * 70)
    pred, true = df["ddg_fwd"].to_numpy(), df["ddg_true"].to_numpy()
    print(f"  mean predicted ddG = {pred.mean():+.3f}   mean true ddG = {true.mean():+.3f}")
    # stabilizing = ddg < thr  (more negative = more stabilizing, kcal/mol convention:
    # NOTE: confirm your sign convention. Here we assume ddg<0 == stabilizing.)
    true_stab = true < thr
    pred_stab = pred < thr
    n_true_stab = int(true_stab.sum())
    n_pred_stab = int(pred_stab.sum())
    tp = int((true_stab & pred_stab).sum())
    ppv = tp / n_pred_stab if n_pred_stab else float("nan")
    recall = tp / n_true_stab if n_true_stab else float("nan")
    print(f"  true stabilizing (ddg<{thr}): {n_true_stab}  "
          f"({100*n_true_stab/len(df):.1f}% of data)")
    print(f"  predicted stabilizing:        {n_pred_stab}")
    print(f"  stabilizing PPV = {ppv:.3f}   recall = {recall:.3f}")
    print("  (Low PPV / few predicted-stabilizing = destabilization bias present)")
    return {"mean_pred": pred.mean(), "mean_true": true.mean(),
            "stab_ppv": ppv, "stab_recall": recall,
            "frac_true_stab": n_true_stab/len(df)}


def q2_scale(df):
    print("\n" + "=" * 70)
    print("Q2 — Do gain and loss occupy different effective scales?")
    print("=" * 70)
    g, l = df["gain_fwd"].to_numpy(), df["loss_fwd"].to_numpy()
    def desc(x):
        return (f"std={x.std():.3f}  IQR={np.percentile(x,75)-np.percentile(x,25):.3f}  "
                f"|.|p95={np.percentile(np.abs(x),95):.3f}  max|.|={np.abs(x).max():.3f}")
    print(f"  gain: {desc(g)}")
    print(f"  loss: {desc(l)}")
    # are large-gain events rarer than large-loss events?
    thr = 1.0
    print(f"  |gain|>{thr}: {(np.abs(g)>thr).mean()*100:.1f}%   "
          f"|loss|>{thr}: {(np.abs(l)>thr).mean()*100:.1f}%")
    ratio = g.std() / l.std() if l.std() else float("nan")
    print(f"  gain/loss std ratio = {ratio:.3f}  "
          f"({'gain compressed' if ratio<0.9 else 'loss compressed' if ratio>1.1 else 'comparable'})")
    return {"gain_std": g.std(), "loss_std": l.std(), "std_ratio": ratio}


def q3_error_attribution(df, thr):
    print("\n" + "=" * 70)
    print("Q3 — On stabilizing mutations, does error track the gain side?")
    print("=" * 70)
    err = (df["ddg_fwd"] - df["ddg_true"]).to_numpy()
    ag, al = df["gain_fwd"].abs().to_numpy(), df["loss_fwd"].abs().to_numpy()
    for label, mask in [("stabilizing (true<%.1f)" % thr, df["ddg_true"] < thr),
                        ("destabilizing", df["ddg_true"] >= thr)]:
        m = mask.to_numpy()
        if m.sum() < 10:
            print(f"  {label}: n={int(m.sum())} too few"); continue
        rg, _ = spearmanr(np.abs(err[m]), ag[m])
        rl, _ = spearmanr(np.abs(err[m]), al[m])
        mean_signed = err[m].mean()
        print(f"  {label} (n={int(m.sum())}): mean signed err={mean_signed:+.3f}  "
              f"|err|~|gain| rho={rg:+.3f}   |err|~|loss| rho={rl:+.3f}")
    print("  (If stabilizing mutations have positive mean error [over-predict ddG,")
    print("   i.e. under-predict stabilization] AND error tracks gain more than")
    print("   loss, that supports the gain-under-calibration hypothesis.)")


def q4_scale_match(df, thr):
    print("\n" + "=" * 70)
    print("Q4 — Counterfactual: rescale gain to match loss spread. Does")
    print("     stabilizing recovery improve without hurting correlation?")
    print("=" * 70)
    g, l = df["gain_fwd"].to_numpy(), df["loss_fwd"].to_numpy()
    true = df["ddg_true"].to_numpy()
    base_pred = g - l
    # rescale gain to match loss std (centered on gain mean to preserve location)
    if g.std() > 0:
        g_rescaled = g.mean() + (g - g.mean()) * (l.std() / g.std())
    else:
        g_rescaled = g
    cf_pred = g_rescaled - l

    def stats(pred, tag):
        rho, _ = spearmanr(pred, true)
        true_stab = true < thr
        pred_stab = pred < thr
        tp = (true_stab & pred_stab).sum()
        ppv = tp / pred_stab.sum() if pred_stab.sum() else float("nan")
        rec = tp / true_stab.sum() if true_stab.sum() else float("nan")
        print(f"  {tag}: overall rho={rho:.3f}  stab PPV={ppv:.3f}  stab recall={rec:.3f}  "
              f"n_pred_stab={int(pred_stab.sum())}")
        return rho, ppv, rec

    b = stats(base_pred, "baseline (g - l)     ")
    c = stats(cf_pred, "gain-rescaled (g'- l)")
    print("\n  Interpretation:")
    print("   - If gain-rescaled IMPROVES stab PPV/recall while overall rho holds,")
    print("     a scale-coupling algorithm is promising -> BUILD IT.")
    print("   - If rho drops sharply or stab recovery doesn't improve, the simple")
    print("     scale story is insufficient -> the algorithm needs a different core,")
    print("     or the bias isn't gain-localized. Report honestly.")
    return {"base_rho": b[0], "base_ppv": b[1], "cf_rho": c[0], "cf_ppv": c[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--stab_threshold", type=float, default=-0.5,
                    help="ddG below this = stabilizing (confirm your sign convention!)")
    ap.add_argument("--out", default="stab_pilot_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading diagnostic CSVs...")
    frames = load_all(args.csvs)

    # Report per-seed and pooled
    summary = []
    for i, df in enumerate(frames):
        print("\n" + "#" * 70)
        print(f"# SEED/FILE {i}: {df['_src'].iloc[0]}   (n={len(df)})")
        print("#" * 70)
        s1 = q1_bias(df, args.stab_threshold)
        s2 = q2_scale(df)
        q3_error_attribution(df, args.stab_threshold)
        s4 = q4_scale_match(df, args.stab_threshold)
        row = {"file": df["_src"].iloc[0], **s1, **s2, **s4}
        summary.append(row)

    pd.DataFrame(summary).to_csv(os.path.join(args.out, "pilot_summary.csv"), index=False)
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print("""
  Build the scale-coupling algorithm IF, consistently across seeds:
    Q1 shows real destabilization bias (low stab PPV),
    Q2 shows gain is compressed relative to loss (std_ratio < ~0.9),
    Q3 shows stabilizing-mutation error tracks the gain component,
    Q4 shows gain-rescaling improves stab recovery without killing rho.

  If these DON'T line up, the elegant hypothesis is wrong. That is a real
  finding too, but it means we do NOT build the scale-coupling method — we
  either find the true source of the bias or drop the direction. Do not
  rationalize a null result into a green light.
""")
    print(f"Saved {args.out}/pilot_summary.csv")


if __name__ == "__main__":
    main()

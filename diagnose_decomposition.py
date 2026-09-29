"""
diagnose_decomposition.py — Test Hypothesis C
==============================================

Question: Are "gain" and "loss" physically meaningful, or are they just
two arbitrary learned scalars A and B where ΔΔG = A - B?

If the decomposition is REAL, these predictions should hold:

TEST 1 — Identity / near-neutral mutations
    For mutations between similar amino acids (high BLOSUM score, e.g.
    L→I, V→I, D→E), both gain and loss should be SMALL in magnitude,
    because little is gained or lost. A spurious A-B decomposition has
    no reason to produce small A and small B here.

TEST 2 — Loss tracks wildtype burial / structural importance
    The "loss" component should correlate with how structurally important
    the wildtype residue is. Buried hydrophobic residues (high contact
    number) should have LARGER loss when mutated away. We test correlation
    between loss magnitude and the number of Cα neighbors within 8Å.

TEST 3 — Reverse-mutation consistency (the strongest test)
    For a mutation A→B at a site, the decomposition says:
        ΔΔG(A→B) = gain(B | context) - loss(A | context)
    For the reverse B→A at the SAME site (same structural context):
        ΔΔG(B→A) = gain(A | context) - loss(B | context)
    If gain and loss are physically real and context-anchored, then the
    gain of residue X in forward should relate to the loss of residue X
    in reverse. Specifically, we test whether:
        gain(B)_fwd  correlates with  loss(B)_rev
    A spurious decomposition has no reason to satisfy this.

TEST 4 — Substitution symmetry of the embedding
    The learned 20×20 mutation-type embedding: do (wt→mut) and (mut→wt)
    embeddings encode opposite-signed information? We check whether the
    learned embedding's first principal component is anti-symmetric:
        E[i,j] ≈ -E[j,i]  (in the dominant direction)

Usage:
    python diagnose_decomposition.py \\
        --data_root /path/to/data_root \\
        --ckpt runs/delta_full/best.pt \\
        --out runs/delta_full/diagnosis
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from protstab_data import Alphabet, Featurizer, MegaScaleTestDatasets
from protstab_models.delta import DELTA, DELTAConfig
from protstab_models.decomposition import DecompositionConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# BLOSUM62 self-vs-substitution: higher = more conservative substitution
BLOSUM62 = {
    ("L", "I"): 2, ("I", "V"): 3, ("V", "I"): 3, ("D", "E"): 2, ("E", "D"): 2,
    ("K", "R"): 2, ("R", "K"): 2, ("F", "Y"): 3, ("Y", "F"): 3, ("S", "T"): 1,
    ("L", "M"): 2, ("M", "L"): 2, ("Q", "E"): 2, ("N", "D"): 1, ("A", "S"): 1,
    ("L", "V"): 1, ("I", "L"): 2, ("F", "W"): 1, ("W", "F"): 1,
}


def _move_batch(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def _onehot_to_idx(onehot):
    return onehot.argmax(dim=-1).clamp(max=19)


def _swap_batch(batch):
    """Swap WT/MUT in append_tensors for reverse-mutation analysis."""
    swapped = dict(batch)
    at = batch["append_tensors"]
    if at.size(-1) == 42:
        swapped["append_tensors"] = torch.cat([at[..., 21:], at[..., :21]], dim=-1)
    return swapped


def load_model(ckpt_path, data_root, device, bidirectional=False, **head_kwargs):
    decomp_cfg = DecompositionConfig(d_model=1408, **head_kwargs)
    cfg = DELTAConfig()
    cfg.adapter_layer_indices = [-3]
    cfg.mlp.input_dim = 128
    cfg.mlp.hidden_dim = [512, 512, 512]
    cfg.mlp.output_dim = 21
    cfg.decomposition = decomp_cfg
    cfg.bidirectional = bidirectional
    model = DELTA(cfg).to(device)

    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:
        state = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def collect_decomposition(model, loader, device):
    """Run forward + reverse passes, collect gain/loss/context info."""
    records = []
    for batch in tqdm(loader, desc="Collecting", leave=False):
        batch = _move_batch(batch, device)

        # Forward pass
        ddg_f, gain_f, loss_f = model(batch, return_decomposition=True)
        # Reverse pass (swap wt/mut)
        ddg_r, gain_r, loss_r = model(_swap_batch(batch), return_decomposition=True)

        at = batch["append_tensors"]
        wt_idx = _onehot_to_idx(at[:, :21].float()).cpu().numpy()
        mut_idx = _onehot_to_idx(at[:, 21:].float()).cpu().numpy()

        # Compute Cα contact number at each mutation site (structural importance)
        ca = batch["X"][:, :, 1, :][0]  # (L, 3)
        dist = torch.cdist(ca.unsqueeze(0), ca.unsqueeze(0)).squeeze(0)  # (L,L)
        contacts_8A = (dist < 8.0).sum(dim=-1).float() - 1  # exclude self
        mut_ids = batch["mut_ids"]
        if not isinstance(mut_ids, torch.Tensor):
            mut_ids = torch.tensor(mut_ids)
        mut_ids = mut_ids.long().cpu().numpy()
        contact_nums = contacts_8A.cpu().numpy()[mut_ids]

        # Protein identifier (for case studies); batch carries "name"
        prot_name = batch.get("name", None)
        if isinstance(prot_name, (list, tuple)):
            prot_name = prot_name[0] if prot_name else None

        n = len(wt_idx)
        for i in range(n):
            records.append({
                "protein": prot_name,
                "position": int(mut_ids[i]),
                "wt": ALPHABET[wt_idx[i]],
                "mut": ALPHABET[mut_idx[i]],
                "ddg_fwd": float(ddg_f[i]), "gain_fwd": float(gain_f[i]), "loss_fwd": float(loss_f[i]),
                "ddg_rev": float(ddg_r[i]), "gain_rev": float(gain_r[i]), "loss_rev": float(loss_r[i]),
                "ddg_true": float(batch["ddG"].view(-1)[i]),
                "contacts": float(contact_nums[i]),
            })
    return pd.DataFrame(records)


def test_1_neutral_mutations(df):
    """Conservative substitutions should have small gain AND small loss."""
    print("\n" + "=" * 70)
    print("TEST 1: Do conservative substitutions have small gain & loss?")
    print("=" * 70)

    df["is_conservative"] = df.apply(
        lambda r: (r["wt"], r["mut"]) in BLOSUM62, axis=1
    )
    cons = df[df["is_conservative"]]
    noncons = df[~df["is_conservative"]]

    if len(cons) < 10:
        print("  Not enough conservative-substitution samples.")
        return

    cons_mag = (cons["gain_fwd"].abs() + cons["loss_fwd"].abs()).mean()
    noncons_mag = (noncons["gain_fwd"].abs() + noncons["loss_fwd"].abs()).mean()

    print(f"  Conservative subs (n={len(cons)}):     mean |gain|+|loss| = {cons_mag:.4f}")
    print(f"  Non-conservative subs (n={len(noncons)}): mean |gain|+|loss| = {noncons_mag:.4f}")
    print(f"  Ratio (cons/noncons): {cons_mag/noncons_mag:.3f}")
    if cons_mag < noncons_mag:
        print("  ✓ PASS: conservative substitutions show smaller decomposition magnitude")
    else:
        print("  ✗ FAIL: no magnitude difference — decomposition may not be physical")


def test_2_loss_tracks_burial(df):
    """Loss magnitude should correlate with wildtype structural importance."""
    print("\n" + "=" * 70)
    print("TEST 2: Does loss magnitude track wildtype burial (contact number)?")
    print("=" * 70)

    valid = df[df["contacts"] > 0]
    if len(valid) < 20:
        print("  Not enough samples with contact info.")
        return

    rho_loss, p_loss = spearmanr(valid["loss_fwd"].abs(), valid["contacts"])
    rho_gain, p_gain = spearmanr(valid["gain_fwd"].abs(), valid["contacts"])

    print(f"  Spearman(|loss|, contacts) = {rho_loss:.3f} (p={p_loss:.1e})")
    print(f"  Spearman(|gain|, contacts) = {rho_gain:.3f} (p={p_gain:.1e})")
    print(f"  (Loss should track burial MORE than gain if semantics are real)")
    if rho_loss > rho_gain and rho_loss > 0.1:
        print("  ✓ PASS: loss tracks structural importance more than gain")
    else:
        print("  ~ INCONCLUSIVE: loss does not preferentially track burial")


def test_3_reverse_consistency(df):
    """gain(X)_fwd should relate to loss(X)_rev for the same residue X."""
    print("\n" + "=" * 70)
    print("TEST 3: Reverse-mutation consistency (STRONGEST test)")
    print("=" * 70)
    print("  For A→B: gain is for B. For B→A (same site): loss is for B.")
    print("  If physical, gain(B)_fwd should correlate with loss(B)_rev.")

    # In forward A→B, gain_fwd is gain of B.
    # In reverse B→A (swapped), loss_rev is loss of B (the new "wt" after swap).
    rho, p = spearmanr(df["gain_fwd"], df["loss_rev"])
    rho2, p2 = spearmanr(df["loss_fwd"], df["gain_rev"])

    print(f"  Spearman(gain_fwd, loss_rev) = {rho:.3f} (p={p:.1e})")
    print(f"  Spearman(loss_fwd, gain_rev) = {rho2:.3f} (p={p2:.1e})")
    if abs(rho) > 0.3 and abs(rho2) > 0.3:
        print("  ✓ PASS: forward/reverse components are linked — semantics are anchored")
    elif abs(rho) > 0.15 or abs(rho2) > 0.15:
        print("  ~ PARTIAL: some linkage, but weak")
    else:
        print("  ✗ FAIL: gain/loss are independent across direction — likely spurious A-B")


def test_4_embedding_antisymmetry(model, out_dir):
    """Is the learned mutation-type embedding anti-symmetric: E[i,j] ≈ -E[j,i]?"""
    print("\n" + "=" * 70)
    print("TEST 4: Mutation-type embedding anti-symmetry")
    print("=" * 70)

    E = model.decomp_head.mut_type_embed.embed.detach().cpu().numpy()  # (20,20,d)

    # For each latent dim, measure anti-symmetry: corr(E[i,j], -E[j,i])
    antisym_scores = []
    for d in range(E.shape[2]):
        M = E[:, :, d]
        upper = M[np.triu_indices(20, k=1)]
        lower = M.T[np.triu_indices(20, k=1)]
        if upper.std() > 1e-6 and lower.std() > 1e-6:
            r, _ = pearsonr(upper, -lower)
            antisym_scores.append(r)

    antisym_scores = np.array(antisym_scores)
    mean_antisym = np.nanmean(antisym_scores)
    n_antisym = (antisym_scores > 0.3).sum()

    print(f"  Mean anti-symmetry correlation across {len(antisym_scores)} dims: {mean_antisym:.3f}")
    print(f"  Dimensions with strong anti-symmetry (r>0.3): {n_antisym}/{len(antisym_scores)}")
    if mean_antisym > 0.2 or n_antisym > 5:
        print("  ✓ PASS: embedding encodes directional (anti-symmetric) substitution info")
    else:
        print("  ~ INCONCLUSIVE: embedding is mostly symmetric (BLOSUM-like, not directional)")

    np.save(os.path.join(out_dir, "embedding_antisymmetry_per_dim.npy"), antisym_scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="megascale_test",
                   help="Which benchmark to analyze")
    p.add_argument("--d_sub", type=int, default=64)
    p.add_argument("--d_hidden", type=int, default=512)
    p.add_argument("--k_neighbors", type=int, default=32)
    p.add_argument("--bidirectional", action="store_true",
                   help="Load a DELTA-B (bidirectional) checkpoint")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Loading model from {args.ckpt}")
    model = load_model(
        args.ckpt, args.data_root, device,
        bidirectional=args.bidirectional,
        d_sub=args.d_sub, d_hidden=args.d_hidden, k_neighbors=args.k_neighbors,
    )

    # Load the analysis dataset
    alphabet = Alphabet(name="esm")
    featurizer = Featurizer(alphabet)
    collection = MegaScaleTestDatasets(data_root=args.data_root)

    target_loader = None
    for name, dataset in collection.iter_named():
        if name == args.dataset:
            target_loader = DataLoader(
                dataset, batch_size=1, shuffle=False,
                num_workers=2, collate_fn=featurizer,
            )
            break

    if target_loader is None:
        log.error(f"Dataset {args.dataset} not found")
        return

    log.info(f"Collecting decomposition on {args.dataset}...")
    df = collect_decomposition(model, target_loader, device)
    df.to_csv(os.path.join(args.out, f"decomposition_{args.dataset}.csv"), index=False)
    log.info(f"Collected {len(df)} mutations")

    # Sanity: does gain - loss = ddg?
    recon_err = (df["gain_fwd"] - df["loss_fwd"] - df["ddg_fwd"]).abs().mean()
    log.info(f"Reconstruction check |gain-loss-ddg| = {recon_err:.6f} (should be ~0)")

    # Run all tests
    test_1_neutral_mutations(df)
    test_2_loss_tracks_burial(df)
    test_3_reverse_consistency(df)
    test_4_embedding_antisymmetry(model, args.out)

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
  If TEST 1 & 3 PASS  → decomposition is physically meaningful (Hyp C refuted,
                         the gain/loss story is real and paper-worthy)
  If TEST 3 FAILS     → gain/loss are likely arbitrary A,B (Hyp C confirmed).
                         The interpretability claim needs to be tempered, and
                         the contribution becomes the architecture, not the
                         physical decomposition.
  TEST 2 & 4 are supporting evidence, not decisive on their own.
""")


if __name__ == "__main__":
    main()

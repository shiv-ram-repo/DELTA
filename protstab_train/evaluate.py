"""
evaluate
========
Evaluation utilities. Auto-detects K50-head and MoveB-head checkpoints
(via state-dict key introspection) and swaps the head before loading
weights, so the same evaluator works for all three configurations
(SPURS MLP, K50, Move B) transparently.

For Move B checkpoints, the evaluator can optionally apply the Move B
batch augmentor that attaches per-mutation δ features to each batch.
Pass `move_b_augmentor=<MoveBFeatureAugmentor instance>` to enable this
during evaluation. Without it, the head silently treats the δ channel
as zeros (= equivalent to evaluating the Move B-trained head without
the actual Move B signal — useful for ablation).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from protstab_data import (
    Alphabet,
    Featurizer,
    MegaScaleTestDatasets,
)

from protstab_models import (
    SPURS,
    SPURSConfig,
)

from .metrics       import compute_metrics
from .tta_inference import tta_predict
from .k50_head      import K50Head, K50HeadConfig
from .move_b_features import MoveBHead, MoveBHeadConfig, MoveBFeatureAugmentor

log = logging.getLogger(__name__)


def _move_batch(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def _detect_head_type(state_dict) -> str:
    """Returns one of: 'spurs_mlp', 'k50', 'move_b'."""
    if not isinstance(state_dict, dict):
        return "spurs_mlp"
    keys = list(state_dict.keys())
    if any(k.startswith("mlp.head_t") or k.startswith("mlp.head_c") for k in keys):
        return "k50"
    # Move B head uses the same fcs structure as the SPURS MLP but with
    # a larger first-layer input dimension. We can't distinguish from
    # state_dict shape alone without inspecting it — check fcs[0] shape.
    fcs0_w = state_dict.get("mlp.fcs.0.weight", None)
    if fcs0_w is not None and fcs0_w.shape[1] > 1408:
        return "move_b"
    return "spurs_mlp"


@torch.no_grad()
def evaluate(
    model, loader, device, name: str = "",
    n_aug: int = 1, tta_sigma: float = 0.1,
    move_b_augmentor: Optional[MoveBFeatureAugmentor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on loader and return concatenated (preds, targets)."""
    model.eval()
    preds, targets = [], []
    for batch in tqdm(loader, desc=f"Eval {name}", leave=False):
        # Move B augmentation happens on CPU before device move.
        if move_b_augmentor is not None:
            try:
                batch = move_b_augmentor(batch)
            except KeyError:
                # If the batch is missing required keys (e.g. name for an
                # esoteric test split), skip augmentation rather than crash.
                pass
        batch = _move_batch(batch, device)
        ddg_pred = tta_predict(model, batch, n_aug=n_aug, sigma=tta_sigma)
        ddg_pred = ddg_pred.view(-1).cpu().numpy()
        ddg_true = batch["ddG"].view(-1).cpu().numpy()
        preds.append(ddg_pred)
        targets.append(ddg_true)
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(targets)


def evaluate_all_benchmarks(
    data_root: str,
    ckpt_path: str,
    output_dir: str,
    *,
    proteinmpnn_ckpt: str = "",
    esm_name: str         = "esm2_t33_650M_UR50D",
    adapter_layer: int    = -3,
    tune_mpnn: bool       = True,
    device: Optional[str] = None,
    n_aug: int            = 1,
    tta_sigma: float      = 0.1,
    move_b_cache_path: Optional[str] = None,
    move_b_local_radius: float = 8.0,
) -> pd.DataFrame:
    """
    Evaluate a checkpoint on every test benchmark.

    Auto-detects the head architecture from state_dict keys.
    If `move_b_cache_path` is given AND the checkpoint is a Move B head,
    applies the Move B augmentor at inference.
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Loading model")
    print("=" * 80)

    cfg = SPURSConfig()
    cfg.adapter_layer_indices = [adapter_layer]
    cfg.name                  = esm_name
    cfg.proteinmpnn_ckpt      = proteinmpnn_ckpt
    cfg.encoder.tune          = tune_mpnn
    cfg.mlp.input_dim         = 128
    cfg.mlp.hidden_dim        = [512, 512, 512]
    cfg.mlp.output_dim        = 21
    cfg.mlp.dropout           = 0.1
    model = SPURS(cfg).to(device)

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:
        state = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}

    head_type = _detect_head_type(state)
    print(f"Detected head type: {head_type}")
    head_input_dim = model.mlp.fcs[0].in_features

    if head_type == "k50":
        model.mlp = K50Head(K50HeadConfig(
            input_dim=head_input_dim, hidden_dim=[512, 512, 512],
            n_aa=21, dropout=0.1,
        )).to(device)
    elif head_type == "move_b":
        model.mlp = MoveBHead(MoveBHeadConfig(
            input_dim=head_input_dim,
            delta_feat_dim=MoveBFeatureAugmentor.N_FEAT,
            hidden_dim=[512, 512, 512],
            n_aa=21, dropout=0.1,
        )).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing and len(missing) > 5:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected and len(unexpected) > 5:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    model.eval()

    # Set up Move B augmentor for inference if requested AND head supports it
    move_b_augmentor = None
    if head_type == "move_b" and move_b_cache_path:
        print(f"Loading Move B cache for inference: {move_b_cache_path}")
        move_b_augmentor = MoveBFeatureAugmentor(
            move_b_cache_path, local_radius=move_b_local_radius,
        )
    elif head_type == "move_b" and move_b_cache_path is None:
        print("Move B head detected but no cache path given — "
              "evaluating with zero δ-features (no-Move-B ablation).")

    alphabet   = Alphabet(name="esm")
    featurizer = Featurizer(alphabet)
    collection = MegaScaleTestDatasets(data_root=data_root)

    results = []
    print("\n" + "=" * 80)
    print("Running benchmarks")
    print("=" * 80)
    for name, dataset in collection.iter_named():
        print(f"\nDataset: {name}")
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=2, collate_fn=featurizer,
        )
        preds, targets = evaluate(
            model, loader, device, name=name,
            n_aug=n_aug, tta_sigma=tta_sigma,
            move_b_augmentor=move_b_augmentor,
        )
        metrics = compute_metrics(preds, targets)
        metrics["dataset"] = name
        results.append(metrics)
        print(f"ρ={metrics['spearman']:.4f} | r={metrics['pearson']:.4f} | "
              f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f}")
        np.save(os.path.join(output_dir, f"preds_{name}.npy"),   preds)
        np.save(os.path.join(output_dir, f"targets_{name}.npy"), targets)

    df = pd.DataFrame(results)[
        ["dataset", "n", "spearman", "pearson", "r2", "rmse", "mae"]
    ]
    csv_path = os.path.join(output_dir, "test_metrics.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df)
    print(f"\nSaved metrics to: {csv_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt",      required=True)
    p.add_argument("--out",       required=True)
    p.add_argument("--mpnn_ckpt", default="")
    p.add_argument("--esm_name",  default="esm2_t33_650M_UR50D")
    p.add_argument("--n_aug",     type=int,   default=1)
    p.add_argument("--tta_sigma", type=float, default=0.1)
    p.add_argument("--move_b_cache", default=None,
                   help="Path to delta_cache.pt; only used for Move B checkpoints")
    args = p.parse_args()

    evaluate_all_benchmarks(
        data_root=args.data_root, ckpt_path=args.ckpt,
        output_dir=args.out, proteinmpnn_ckpt=args.mpnn_ckpt,
        esm_name=args.esm_name, n_aug=args.n_aug, tta_sigma=args.tta_sigma,
        move_b_cache_path=args.move_b_cache,
    )

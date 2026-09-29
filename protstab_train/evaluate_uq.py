"""
evaluate_uq.py
==============
Monte Carlo Dropout for Uncertainty Quantification.
Runs multiple forward passes with Dropout enabled to compute a confidence score.
Evaluates how Spearman correlation improves as we filter out uncertain predictions.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

from protstab_data import Alphabet, Featurizer, MegaScaleTestDatasets
from protstab_models import SPURS, SPURSConfig
from protstab_train.k50_head import K50Head, K50HeadConfig

def enable_mc_dropout(model):
    """Force dropout layers to remain active during evaluation."""
    model.eval() # Set everything to eval first (freezes LayerNorm, etc.)
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def _detect_head_type(state_dict) -> str:
    if not isinstance(state_dict, dict):
        return "spurs_mlp"
    return "k50" if any(k.startswith("mlp.head_t") or k.startswith("mlp.head_c") for k in state_dict.keys()) else "spurs_mlp"

def _move_batch(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

@torch.no_grad()
def run_uq_experiment(
    data_root: str,
    ckpt_path: str,
    mpnn_ckpt: str,
    target_dataset: str = "s669",
    n_passes: int = 15,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 80)
    print(f"Loading Model for MC Dropout UQ on {target_dataset}")
    print("=" * 80)

    # 1. Load Model with EXACT Config
    cfg = SPURSConfig()
    cfg.adapter_layer_indices = [-3]
    cfg.name                  = "esm2_t33_650M_UR50D"
    cfg.proteinmpnn_ckpt      = mpnn_ckpt
    cfg.encoder.tune          = True
    cfg.mlp.input_dim         = 128
    cfg.mlp.hidden_dim        = [512, 512, 512]
    cfg.mlp.output_dim        = 21
    cfg.mlp.dropout           = 0.1

    model = SPURS(cfg).to(device)

    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:
        state = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}

    if _detect_head_type(state) == "k50":
        k50_cfg = K50HeadConfig(input_dim=1408, hidden_dim=[512, 512, 512], n_aa=21, dropout=0.1)
        model.mlp = K50Head(k50_cfg).to(device)

    model.load_state_dict(state, strict=False)

    # Enable MC Dropout
    enable_mc_dropout(model)
    print("MC Dropout enabled successfully.")

    # 2. Load Dataset
    alphabet = Alphabet(name="esm")
    featurizer = Featurizer(alphabet)
    collection = MegaScaleTestDatasets(data_root=data_root)

    ds = next((d for n, d in collection.iter_named() if n == target_dataset), None)
    if ds is None:
        raise ValueError(f"Dataset {target_dataset} not found!")

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=featurizer)

    # 3. Run Inference
    results = []

    for batch in tqdm(loader, desc=f"Running UQ ({n_passes} passes/mut)"):
        batch = _move_batch(batch, device)
        ddg_true = batch["ddG"].view(-1).cpu().numpy()

        K = ddg_true.shape[0]

        # Run N passes
        passes = []
        for _ in range(n_passes):
            # Normal forward pass (dropout will randomly mask features)
            pred = model(batch).view(-1).cpu().numpy()
            passes.append(pred)

        passes = np.stack(passes, axis=0) # Shape: (n_passes, K)

        mean_preds = passes.mean(axis=0)
        std_preds = passes.std(axis=0) # This is our UNCERTAINTY score!

        for k in range(K):
            results.append({
                "true": ddg_true[k],
                "pred_mean": mean_preds[k],
                "uncertainty": std_preds[k]
            })

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("UNCERTAINTY FILTERING RESULTS")
    print("=" * 80)

    # 4. Evaluate across confidence thresholds
    # Sort by uncertainty (lowest variance to highest variance)
    df = df.sort_values("uncertainty", ascending=True).reset_index(drop=True)

    thresholds = [1.0, 0.9, 0.8, 0.7, 0.5]

    for t in thresholds:
        keep_n = int(len(df) * t)
        subset = df.iloc[:keep_n]

        spearman, _ = spearmanr(subset["true"], subset["pred_mean"])
        pearson, _ = pearsonr(subset["true"], subset["pred_mean"])

        print(f"Top {int(t*100)}% Confident (N={keep_n:3d}) | Spearman: {spearman:.4f} | Pearson: {pearson:.4f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mpnn_ckpt", required=True)
    args = p.parse_args()

    run_uq_experiment(args.data_root, args.ckpt, args.mpnn_ckpt, target_dataset="s669")

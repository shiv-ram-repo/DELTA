"""
evaluate_with_move_b
====================
Final convergence script: Evaluates a K50 multi-task checkpoint using
Phase 1 precomputed Move B (delta coordinate) tensors from a single .pt file.
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

from protstab_data import Alphabet, Featurizer, MegaScaleTestDatasets
from protstab_models import SPURS, SPURSConfig

from .metrics import compute_metrics
from .tta_inference import tta_predict
from .k50_head import K50Head, K50HeadConfig

log = logging.getLogger(__name__)

def _move_batch(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out

def _detect_head_type(state_dict) -> str:
    if not isinstance(state_dict, dict):
        return "spurs_mlp"
    has_k50 = any(
        k.startswith("mlp.head_t") or k.startswith("mlp.head_c")
        for k in state_dict.keys()
    )
    return "k50" if has_k50 else "spurs_mlp"

@torch.no_grad()
def evaluate_move_b(
    model, loader, device, delta_cache_dict: dict, dataset_name: str,
    n_aug: int = 1, tta_sigma: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []

    missing_deltas = 0
    total_muts = 0

    for batch in tqdm(loader, desc=f"Eval {dataset_name} (Move B)", leave=False):
        batch = _move_batch(batch, device)

        # --- 1. THE PERFECT NAME EXTRACTION ---
        name_raw = batch.get("name", "")
        if isinstance(name_raw, (list, tuple)):
            name_raw = name_raw[0]

        wt_name = str(name_raw)
        if wt_name.endswith(".pdb"):
            wt_name = wt_name[:-4]  # Strip .pdb to match cache

        # --- 2. EXTRACT MUTATION IDS AND AAs SAFELY ---
        mut_ids = batch.get("mut_ids", [])
        if hasattr(mut_ids, "cpu"):
            mut_ids = mut_ids.cpu().view(-1).tolist()

        append_tensors = batch.get("append_tensors")
        mut_aas = []
        if append_tensors is not None:
            mut_aas = append_tensors[..., 21:42].argmax(dim=-1).cpu().view(-1).tolist()

        K = len(mut_ids)
        total_muts += K
        L = batch["X"].shape[1]

        delta_list = []
        has_any_move_b = False

        # --- 3. LOOKUP AND ALIGN TENSORS ---
        for k in range(K):
            pos = int(mut_ids[k])
            aa = int(mut_aas[k]) if k < len(mut_aas) else -1

            key = (wt_name, pos, aa)

            if key in delta_cache_dict:
                d_tensor = delta_cache_dict[key].to(device)
                d_L = d_tensor.shape[0]

                # Fix length mismatch dynamically to avoid stacking crashes!
                if d_L != L:
                    aligned_d = torch.zeros((L, 4, 3), device=device)
                    min_L = min(L, d_L)
                    aligned_d[:min_L] = d_tensor[:min_L]
                    d_tensor = aligned_d

                delta_list.append(d_tensor)
                has_any_move_b = True
            else:
                missing_deltas += 1
                delta_list.append(torch.zeros((L, 4, 3), device=device))

        # Inject the stacked deltas into the batch
        if has_any_move_b:
            batch["delta_coords"] = torch.stack(delta_list, dim=0) # (K, L, 4, 3)
            batch["use_move_b"] = True
        else:
            batch["use_move_b"] = False
        # ------------------------------------------------

        ddg_pred = tta_predict(model, batch, n_aug=n_aug, sigma=tta_sigma)
        ddg_pred = ddg_pred.view(-1).cpu().numpy()
        ddg_true = batch["ddG"].view(-1).cpu().numpy()

        preds.append(ddg_pred)
        targets.append(ddg_true)

    if missing_deltas > 0:
        log.warning(f"[{dataset_name}] Missing {missing_deltas}/{total_muts} delta tensors. Used WT fallback.")

    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(targets)


def evaluate_all_with_move_b(
    data_root: str,
    ckpt_path: str,
    delta_file: str,
    output_dir: str,
    *,
    proteinmpnn_ckpt: str = "",
    esm_name: str         = "esm2_t33_650M_UR50D",
    adapter_layer: int    = -3,
    tune_mpnn: bool       = True,
    device: Optional[str] = None,
) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Loading Model & K50 Head for Move B Evaluation")
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
    if head_type == "k50":
        head_input_dim = model.mlp.fcs[0].in_features
        k50_cfg = K50HeadConfig(
            input_dim=head_input_dim,
            hidden_dim=[512, 512, 512],
            n_aa=21, dropout=0.1,
        )
        model.mlp = K50Head(k50_cfg).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()

    print(f"\nLoading Move B Delta Cache from: {delta_file}")
    delta_cache_dict = torch.load(delta_file, map_location="cpu")
    print(f"Cache loaded successfully. Contains {len(delta_cache_dict):,} mutations.")

    alphabet   = Alphabet(name="esm")
    featurizer = Featurizer(alphabet)
    collection = MegaScaleTestDatasets(data_root=data_root)

    results = []
    print("\n" + "=" * 80)
    print("Running benchmarks WITH Delta Cache")
    print("=" * 80)

    for name, dataset in collection.iter_named():
        print(f"\nDataset: {name}")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=featurizer)

        preds, targets = evaluate_move_b(
            model, loader, device,
            delta_cache_dict=delta_cache_dict,
            dataset_name=name,
            n_aug=1, tta_sigma=0.1
        )

        metrics = compute_metrics(preds, targets)
        metrics["dataset"] = name
        results.append(metrics)
        print(f"ρ={metrics['spearman']:.4f} | r={metrics['pearson']:.4f} | RMSE={metrics['rmse']:.4f}")

    df = pd.DataFrame(results)[["dataset", "n", "spearman", "pearson", "r2", "rmse", "mae"]]
    csv_path = os.path.join(output_dir, "move_b_test_metrics.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")

    print("\n" + "=" * 80)
    print("FINAL MOVE B RESULTS")
    print("=" * 80)
    print(df)
    return df

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt",      required=True)
    p.add_argument("--delta_file", required=True, help="Path to your Phase 1 .pt file")
    p.add_argument("--out",       required=True)
    p.add_argument("--mpnn_ckpt", default="")
    args = p.parse_args()

    evaluate_all_with_move_b(
        data_root=args.data_root, ckpt_path=args.ckpt,
        delta_file=args.delta_file, output_dir=args.out,
        proteinmpnn_ckpt=args.mpnn_ckpt,
    )

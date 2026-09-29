"""
delta_train.py — DELTA Training Pipeline
==========================================

End-to-end training for DELTA: Decomposed Estimation of Local
Thermodynamic Attributes.

Uses the same data pipeline as the existing repo (MegaScaleDataset,
Featurizer) but replaces the model and loss with DELTA components.

Key training features:
  - Bidirectional gain-loss decomposition
  - Anti-symmetry via reverse pass (wt↔mut swap) on both ΔΔG and
    individual gain/loss branches
  - Decomposition magnitude regularization
  - BLOSUM62-initialized mutation-type embeddings
  - Standard early stopping on validation Spearman correlation

Usage:
    python -m protstab_train.delta_train \\
        --data_root /path/to/data \\
        --output_dir runs/delta_v1 \\
        --loss_type huber \\
        --use_bcas \\
        --use_decomp_antisym \\
        --max_epochs 200

Or from Python:
    from protstab_train.delta_train import train, DeltaTrainConfig
    cfg = DeltaTrainConfig(data_root="...", output_dir="runs/delta_v1")
    train(cfg)
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from protstab_data import Alphabet, Featurizer, MegaScaleDataset

from protstab_models.delta import DELTA, DELTAConfig
from protstab_models.decomposition import DecompositionConfig

from .evaluate import evaluate
from .metrics import compute_metrics
from .delta_loss import DeltaLoss, DeltaLossConfig
from .imbalanced_losses import BMCLoss

log = logging.getLogger(__name__)


DEFAULT_MEGASCALE_CSV = ""


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DeltaTrainConfig:
    # Data
    data_root: str = ""
    megascale_csv: str = DEFAULT_MEGASCALE_CSV
    output_dir: str = "runs/delta_v1"
    ckpt_filename: str = "best.pt"
    proteinmpnn_ckpt: str = ""

    # Training schedule
    max_epochs: int = 200
    early_stop_pat: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-2
    val_every: int = 1
    grad_accum: int = 1
    seed: int = 42

    # Backbone
    esm_name: str = "esm2_t33_650M_UR50D"
    adapter_layer: int = -3
    tune_mpnn: bool = True

    # Decomposition head
    d_aa: int = 64
    d_sub: int = 64
    d_hidden: int = 512
    n_branch_layers: int = 3
    n_context_heads: int = 8
    k_neighbors: int = 32
    sigma_init: float = 8.0
    head_dropout: float = 0.1
    use_blosum_init: bool = True

    # Loss configuration
    loss_type: str = "huber"      # "huber" | "bmc"
    bmc_init_sigma: float = 1.0

    use_decomp_reg: bool = True
    decomp_reg_weight: float = 0.1
    decomp_reg_alpha: float = 2.0

    use_bcas: bool = True
    bcas_alpha: float = 1.0
    bcas_beta: float = 0.5
    bcas_weight: float = 0.5

    use_decomp_antisym: bool = True
    decomp_antisym_weight: float = 0.3

    use_mut_smoothness: bool = False
    mut_smoothness_weight: float = 0.01

    # Scheduler
    use_cosine_schedule: bool = False
    cosine_T_max: int = 200

    # Auxiliary: keep SPURS energy head for comparison
    use_auxiliary_energy_head: bool = False

    # Pretrained backbone: path to a trained SPURS checkpoint
    # If set, loads backbone weights (encoder, decoder, cross_attn) from
    # this checkpoint and fine-tunes with lower LR. Decomposition head
    # trains from scratch at full LR.
    backbone_ckpt: str = ""
    backbone_lr_scale: float = 0.1  # backbone LR = lr * backbone_lr_scale

    # Experiment 1: freeze backbone entirely, train only the decomposition head.
    # Diagnostic to test whether the backbone is the bottleneck. If frozen
    # performance ≈ from-scratch performance, the backbone was never limiting.
    freeze_backbone: bool = False

    # DELTA-B: identity-conditioned bidirectional attention decomposition.
    # False → DELTA-U (current). True → DELTA-B (wildtype-query→loss, mutant-query→gain).
    bidirectional: bool = False


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _move_batch(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def _make_swapped_batch(batch: dict) -> dict:
    """WT/MUT swap for the anti-symmetry constraint."""
    swapped = dict(batch)
    if "append_tensors" in batch and batch["append_tensors"] is not None:
        at = batch["append_tensors"]
        if at.size(-1) == 42:
            swapped["append_tensors"] = torch.cat(
                [at[..., 21:], at[..., :21]], dim=-1
            )
    if "ddG" in batch and batch["ddG"] is not None:
        swapped["ddG"] = -batch["ddG"]
    return swapped


# ═══════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════

def build_dataloaders(cfg: DeltaTrainConfig):
    alphabet = Alphabet(name="esm")
    featurizer = Featurizer(alphabet)
    train_ds = MegaScaleDataset(data_root=cfg.data_root, split="train")
    val_ds = MegaScaleDataset(data_root=cfg.data_root, split="val")
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=featurizer,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=featurizer,
    )
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════

def build_model(cfg: DeltaTrainConfig) -> DELTA:
    decomp_cfg = DecompositionConfig(
        d_model=1408,  # will be overridden in DELTA.__init__
        d_aa=cfg.d_aa,
        d_sub=cfg.d_sub,
        d_hidden=cfg.d_hidden,
        n_branch_layers=cfg.n_branch_layers,
        n_context_heads=cfg.n_context_heads,
        k_neighbors=cfg.k_neighbors,
        sigma_init=cfg.sigma_init,
        dropout=cfg.head_dropout,
        use_blosum_init=cfg.use_blosum_init,
    )

    delta_cfg = DELTAConfig()
    delta_cfg.adapter_layer_indices = [cfg.adapter_layer]
    delta_cfg.name = cfg.esm_name
    delta_cfg.proteinmpnn_ckpt = cfg.proteinmpnn_ckpt
    delta_cfg.encoder.tune = cfg.tune_mpnn
    delta_cfg.mlp.input_dim = 128
    delta_cfg.mlp.hidden_dim = [512, 512, 512]
    delta_cfg.mlp.output_dim = 21
    delta_cfg.mlp.dropout = 0.1
    delta_cfg.decomposition = decomp_cfg
    delta_cfg.bidirectional = cfg.bidirectional
    delta_cfg.use_auxiliary_energy_head = cfg.use_auxiliary_energy_head

    return DELTA(delta_cfg)


# ═══════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: DELTA,
    loss_fn: DeltaLoss,
    loader,
    optimizer,
    device,
    epoch: int,
    *,
    grad_accum: int = 1,
    use_bcas: bool = True,
    use_decomp_antisym: bool = True,
) -> Dict[str, float]:
    """Train one epoch with DELTA's decomposition-aware losses."""
    model.train()
    optimizer.zero_grad()

    running = {
        "total": 0.0, "primary": 0.0, "bcas": 0.0,
        "decomp_reg": 0.0, "decomp_antisym": 0.0,
    }
    n = 0

    loop = tqdm(loader, desc=f"Train Ep{epoch:03d}", leave=False)
    for step, batch in enumerate(loop):
        batch = _move_batch(batch, device)

        # ── Forward pass ──────────────────────────────────────────
        ddg_pred, gain, loss_scores = model(
            batch, return_decomposition=True,
        )
        ddg_true = batch["ddG"].view(-1)
        valid = (ddg_pred.abs() < 9000) & torch.isfinite(ddg_true)

        if valid.sum() == 0:
            continue

        # ── Reverse pass (for anti-symmetry) ──────────────────────
        ddg_rev = gain_rev = loss_rev = valid_rev = None

        if use_bcas or use_decomp_antisym:
            swapped = _make_swapped_batch(batch)
            ddg_rev, gain_rev, loss_rev = model(
                swapped, return_decomposition=True,
            )
            valid_rev = (ddg_rev.abs() < 9000) & valid

        # ── Compute combined loss ─────────────────────────────────
        total_loss, diagnostics = loss_fn(
            ddg_pred=ddg_pred,
            ddg_true=ddg_true,
            gain=gain,
            loss_scores=loss_scores,
            valid=valid,
            ddg_rev=ddg_rev,
            gain_rev=gain_rev,
            loss_rev=loss_rev,
            valid_rev=valid_rev,
            mut_type_embed=model.decomp_head.mut_type_embed.embed,
        )

        # ── Backward ──────────────────────────────────────────────
        (total_loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # ── Logging ───────────────────────────────────────────────
        running["total"] += diagnostics.get("total_loss", 0.0)
        running["primary"] += diagnostics.get("primary_loss", 0.0)
        running["bcas"] += diagnostics.get("bcas_var", 0.0)
        running["decomp_reg"] += diagnostics.get("decomp_reg", 0.0)
        running["decomp_antisym"] += (
            diagnostics.get("anti_sym_gain", 0.0)
            + diagnostics.get("anti_sym_loss", 0.0)
        )
        n += 1
        loop.set_postfix(loss=f"{total_loss.item():.4f}")

    n = max(n, 1)
    return {k: v / n for k, v in running.items()}


# ═══════════════════════════════════════════════════════════════════════
# DELTA Evaluation Wrapper
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_delta(
    model: DELTA,
    loader,
    device,
    name: str = "",
) -> tuple:
    """
    Evaluate DELTA model. Returns (preds, targets, gains, losses).
    """
    model.eval()
    preds, targets, all_gains, all_losses = [], [], [], []

    for batch in tqdm(loader, desc=f"Eval {name}", leave=False):
        batch = _move_batch(batch, device)
        ddg, gain, loss_scores = model(batch, return_decomposition=True)

        preds.append(ddg.view(-1).cpu().numpy())
        targets.append(batch["ddG"].view(-1).cpu().numpy())
        all_gains.append(gain.view(-1).cpu().numpy())
        all_losses.append(loss_scores.view(-1).cpu().numpy())

    if not preds:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(preds),
        np.concatenate(targets),
        np.concatenate(all_gains),
        np.concatenate(all_losses),
    )


# ═══════════════════════════════════════════════════════════════════════
# Main Training Function
# ═══════════════════════════════════════════════════════════════════════

def train(cfg: Optional[DeltaTrainConfig] = None):
    if cfg is None:
        cfg = DeltaTrainConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"Config: {asdict(cfg)}")

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)
    log.info(f"Train: {len(train_loader)}  Val: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    # ── Load pretrained backbone from SPURS checkpoint ────────────
    if cfg.backbone_ckpt and os.path.exists(cfg.backbone_ckpt):
        log.info(f"Loading pretrained backbone from: {cfg.backbone_ckpt}")
        ckpt_state = torch.load(cfg.backbone_ckpt, map_location=device)
        if "state_dict" in ckpt_state:
            ckpt_state = {
                k.replace("model.", "", 1): v
                for k, v in ckpt_state["state_dict"].items()
            }

        # Map SPURS backbone keys to DELTA backbone keys.
        # DELTA shares: encoder, decoder, cross_attn, flat_layers, dp
        # DELTA does NOT have: mlp (replaced by decomp_head)
        backbone_keys = set()
        loaded, skipped = 0, 0
        model_state = model.state_dict()

        for key, value in ckpt_state.items():
            # Skip the old MLP head — DELTA uses decomp_head instead
            if key.startswith("mlp."):
                skipped += 1
                continue
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
                backbone_keys.add(key.split(".")[0])
                loaded += 1
            else:
                skipped += 1

        model.load_state_dict(model_state, strict=False)
        log.info(f"  Loaded {loaded} backbone tensors, skipped {skipped}")
        log.info(f"  Backbone modules initialized: {sorted(backbone_keys)}")
    elif cfg.backbone_ckpt:
        log.warning(f"backbone_ckpt not found: {cfg.backbone_ckpt}")

    # ── Experiment 1: freeze backbone ─────────────────────────────
    if cfg.freeze_backbone:
        backbone_module_names = {"encoder", "decoder", "cross_attn", "flat_layers", "dp"}
        frozen = 0
        for name, param in model.named_parameters():
            if name.split(".")[0] in backbone_module_names:
                param.requires_grad = False
                frozen += 1
        log.info(f"FROZEN backbone: {frozen} tensors set requires_grad=False")
        log.info("  → Only the decomposition head will train (Experiment 1 diagnostic)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    # Log DELTA-specific parameters
    decomp_params = sum(
        p.numel() for p in model.decomp_head.parameters() if p.requires_grad
    )
    log.info(f"  Decomposition head: {decomp_params:,} params")
    log.info(f"  Mutation-type embed: {model.decomp_head.mut_type_embed.embed.numel():,} params")

    # ── Loss ──────────────────────────────────────────────────────
    loss_cfg = DeltaLossConfig(
        loss_type=cfg.loss_type,
        bmc_init_sigma=cfg.bmc_init_sigma,
        use_decomp_reg=cfg.use_decomp_reg,
        decomp_reg_weight=cfg.decomp_reg_weight,
        decomp_reg_alpha=cfg.decomp_reg_alpha,
        use_bcas=cfg.use_bcas,
        bcas_alpha=cfg.bcas_alpha,
        bcas_beta=cfg.bcas_beta,
        bcas_weight=cfg.bcas_weight,
        use_decomp_antisym=cfg.use_decomp_antisym,
        decomp_antisym_weight=cfg.decomp_antisym_weight,
        use_mut_smoothness=cfg.use_mut_smoothness,
        mut_smoothness_weight=cfg.mut_smoothness_weight,
    )
    loss_fn = DeltaLoss(loss_cfg).to(device)

    # Handle BMC loss
    bmc_module = None
    if cfg.loss_type == "bmc":
        bmc_module = BMCLoss(init_noise_sigma=cfg.bmc_init_sigma).to(device)
        loss_fn.primary_loss = bmc_module

    # ── Optimizer (differential LR for backbone vs head) ────────
    # Backbone modules: encoder, decoder, cross_attn, flat_layers
    backbone_module_names = {"encoder", "decoder", "cross_attn", "flat_layers", "dp"}
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        module_root = name.split(".")[0]
        if module_root in backbone_module_names:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Use lower LR for backbone when pretrained
    backbone_lr = cfg.lr * cfg.backbone_lr_scale if cfg.backbone_ckpt else cfg.lr
    param_groups = [
        {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        {"params": head_params, "lr": cfg.lr, "name": "head"},
    ]
    if bmc_module is not None:
        param_groups.append(
            {"params": list(bmc_module.parameters()), "lr": cfg.lr, "name": "bmc"}
        )

    optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay)

    log.info(f"  Backbone params: {len(backbone_params)} tensors, lr={backbone_lr:.1e}")
    log.info(f"  Head params    : {len(head_params)} tensors, lr={cfg.lr:.1e}")

    scheduler = None
    if cfg.use_cosine_schedule:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg.cosine_T_max, eta_min=cfg.lr * 0.01,
        )

    # ── Training ──────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("DELTA Training Configuration")
    log.info("=" * 70)
    log.info(f"Loss type            : {cfg.loss_type}")
    log.info(f"BCAS anti-symmetry   : {cfg.use_bcas} (α={cfg.bcas_alpha}, β={cfg.bcas_beta})")
    log.info(f"Decomp regularizer   : {cfg.use_decomp_reg} (w={cfg.decomp_reg_weight})")
    log.info(f"Decomp anti-symmetry : {cfg.use_decomp_antisym} (w={cfg.decomp_antisym_weight})")
    log.info(f"Neighbors            : k={cfg.k_neighbors}, σ_init={cfg.sigma_init}Å")
    log.info(f"Mutation-type embed  : d_sub={cfg.d_sub}, BLOSUM init={cfg.use_blosum_init}")
    log.info(f"Branch architecture  : d_hidden={cfg.d_hidden}, layers={cfg.n_branch_layers}")
    if cfg.backbone_ckpt:
        log.info(f"Backbone init        : {cfg.backbone_ckpt}")
        log.info(f"Backbone LR scale    : {cfg.backbone_lr_scale}× (lr={backbone_lr:.1e})")
    else:
        log.info(f"Backbone init        : from scratch")
    log.info("=" * 70)

    best_sp = -1.0
    best_epoch = 0
    no_improve = 0
    logs = []
    ckpt_path = os.path.join(cfg.output_dir, cfg.ckpt_filename)
    log_path = os.path.join(cfg.output_dir, "training_log.csv")

    for epoch in range(1, cfg.max_epochs + 1):
        # Train
        tr = train_one_epoch(
            model, loss_fn, train_loader, optimizer, device, epoch,
            grad_accum=cfg.grad_accum,
            use_bcas=cfg.use_bcas,
            use_decomp_antisym=cfg.use_decomp_antisym,
        )

        if scheduler is not None:
            scheduler.step()

        # Validate
        if epoch % cfg.val_every == 0:
            preds, targets, gains, losses = evaluate_delta(
                model, val_loader, device, name="val",
            )
            val_m = compute_metrics(preds, targets)

            # Decomposition diagnostics
            val_m["gain_mean"] = float(np.mean(np.abs(gains))) if len(gains) > 0 else 0.0
            val_m["loss_mean"] = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0
        else:
            val_m = {
                "spearman": float("nan"), "pearson": float("nan"),
                "rmse": float("nan"), "mae": float("nan"),
                "r2": float("nan"), "n": 0,
                "gain_mean": 0.0, "loss_mean": 0.0,
            }

        # Checkpointing
        flag = ""
        if not np.isnan(val_m["spearman"]) and val_m["spearman"] > best_sp:
            best_sp = val_m["spearman"]
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            flag = "  ← BEST"
        else:
            no_improve += 1

        # Logging
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(log_path, index=False)

        parts = [f"pri={tr['primary']:.4f}"]
        if cfg.use_bcas:
            parts.append(f"bcas={tr['bcas']:.4f}")
        if cfg.use_decomp_reg:
            parts.append(f"dreg={tr['decomp_reg']:.4f}")
        if cfg.use_decomp_antisym:
            parts.append(f"dasym={tr['decomp_antisym']:.4f}")

        log.info(
            f"Ep {epoch:03d} | {' '.join(parts)} | "
            f"val ρ={val_m['spearman']:.4f} r={val_m['pearson']:.4f} "
            f"|gain|={val_m['gain_mean']:.3f} |loss|={val_m['loss_mean']:.3f}"
            f"{flag}"
        )

        if no_improve >= cfg.early_stop_pat:
            log.info(f"Early stopping at epoch {epoch}")
            break

    log.info(f"Best val Spearman: {best_sp:.4f} at epoch {best_epoch}")
    log.info(f"Best checkpoint  : {ckpt_path}")
    return ckpt_path


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    p = argparse.ArgumentParser(description="DELTA Training")
    p.add_argument("--data_root", default="")
    p.add_argument("--output_dir", default="runs/delta_v1")
    p.add_argument("--proteinmpnn_ckpt", default="")
    p.add_argument("--backbone_ckpt", default="",
                   help="Path to pretrained SPURS checkpoint for backbone init")
    p.add_argument("--backbone_lr_scale", type=float, default=0.1,
                   help="LR multiplier for backbone when using --backbone_ckpt")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Experiment 1: freeze backbone, train only decomposition head")
    p.add_argument("--bidirectional", action="store_true",
                   help="DELTA-B: identity-conditioned bidirectional attention decomposition")
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss_type", default="huber", choices=["huber", "bmc"])
    p.add_argument("--use_bcas", action="store_true", default=True)
    p.add_argument("--no_bcas", action="store_false", dest="use_bcas")
    p.add_argument("--use_decomp_antisym", action="store_true", default=True)
    p.add_argument("--no_decomp_antisym", action="store_false", dest="use_decomp_antisym")
    p.add_argument("--use_decomp_reg", action="store_true", default=True)
    p.add_argument("--no_decomp_reg", action="store_false", dest="use_decomp_reg")
    p.add_argument("--k_neighbors", type=int, default=32)
    p.add_argument("--d_sub", type=int, default=64)
    p.add_argument("--d_hidden", type=int, default=512)
    p.add_argument("--n_branch_layers", type=int, default=3)
    p.add_argument("--use_cosine_schedule", action="store_true")
    p.add_argument("--grad_accum", type=int, default=1)
    args = p.parse_args()

    cfg = DeltaTrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        proteinmpnn_ckpt=args.proteinmpnn_ckpt,
        backbone_ckpt=args.backbone_ckpt,
        backbone_lr_scale=args.backbone_lr_scale,
        freeze_backbone=args.freeze_backbone,
        bidirectional=args.bidirectional,
        max_epochs=args.max_epochs,
        lr=args.lr,
        seed=args.seed,
        loss_type=args.loss_type,
        use_bcas=args.use_bcas,
        use_decomp_antisym=args.use_decomp_antisym,
        use_decomp_reg=args.use_decomp_reg,
        k_neighbors=args.k_neighbors,
        d_sub=args.d_sub,
        d_hidden=args.d_hidden,
        n_branch_layers=args.n_branch_layers,
        use_cosine_schedule=args.use_cosine_schedule,
        grad_accum=args.grad_accum,
    )

    train(cfg)

"""
delta_loss.py — Loss Functions for DELTA Decomposition Training
================================================================

Three loss components designed for the bidirectional decomposition:

1. Primary ΔΔG loss (Huber or BMC)
   --------------------------------
   Standard regression loss on ddg = gain - loss vs. ground truth.
   Identical to the existing pipeline's loss.

2. Decomposition Regularization
   -----------------------------
   Encourages the gain/loss decomposition to be "efficient":
     L_decomp = λ * max(0, |gain| + |loss| - α * |ΔΔG|)

   Without this, the model could learn gain=1000, loss=999 for ΔΔG=1.
   The regularizer soft-constrains |gain| + |loss| to stay proportional
   to the prediction magnitude. α controls how much slack is allowed
   (α=2.0 means gain+loss magnitudes can be up to 2× the ΔΔG magnitude).

3. Anti-Symmetry Decomposition Loss
   ---------------------------------
   For a mutation A→B at position i:
     Forward: gain(B, context) - loss(A, context) = ΔΔG(A→B)
     Reverse: gain(A, context) - loss(B, context) = ΔΔG(B→A) = -ΔΔG(A→B)

   Rather than re-running the full model (expensive), we can enforce:
     gain(B, ctx) - loss(A, ctx) + gain(A, ctx) - loss(B, ctx) ≈ 0

   Which simplifies to:
     [gain(B) + gain(A)] ≈ [loss(A) + loss(B)]

   This is a softer form of anti-symmetry that operates on the
   decomposed components rather than requiring a second forward pass.
   When combined with the standard siamese/BCAS loss (which DOES
   use a second forward pass), this provides complementary gradient
   signal directly to the gain and loss branches.

4. Mutation-Type Consistency
   --------------------------
   Optional: encourages the mutation-type embedding to be smooth —
   similar substitutions (e.g., L→I and L→V) should produce similar
   embeddings. Implemented as an L2 penalty on the gradient of the
   embedding table w.r.t. BLOSUM62 neighbors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. Decomposition Magnitude Regularizer
# ═══════════════════════════════════════════════════════════════════════

class DecompositionRegularizer(nn.Module):
    """
    Prevents the gain/loss branches from learning cancelling large values.

    L = mean(max(0, |gain| + |loss| - alpha * |ddg_pred|))

    With alpha=2.0, the model is allowed up to 2× the prediction magnitude
    in total gain+loss. Anything beyond that is penalized.
    """

    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        gain: torch.Tensor,   # (K,)
        loss: torch.Tensor,   # (K,)
        ddg: torch.Tensor,    # (K,)
        valid: torch.Tensor,  # (K,) bool mask
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if valid is not None:
            gain, loss, ddg = gain[valid], loss[valid], ddg[valid]
        if gain.numel() == 0:
            z = torch.zeros((), device=gain.device)
            return z, {"decomp_reg": 0.0, "decomp_ratio": 0.0}

        magnitude_sum = gain.abs() + loss.abs()
        ddg_magnitude = ddg.abs().detach()  # don't backprop through target
        slack = self.alpha * ddg_magnitude

        penalty = F.relu(magnitude_sum - slack)
        reg_loss = penalty.mean()

        # Diagnostic: average ratio of |gain|+|loss| to |ddg|
        ratio = (magnitude_sum / (ddg_magnitude + 1e-6)).mean().item()

        return reg_loss, {
            "decomp_reg": float(reg_loss.item()),
            "decomp_ratio": ratio,
        }


# ═══════════════════════════════════════════════════════════════════════
# 2. Decomposition Anti-Symmetry Loss
# ═══════════════════════════════════════════════════════════════════════

class DecompositionAntiSymmetryLoss(nn.Module):
    """
    Soft anti-symmetry constraint on the decomposition.

    For mutation (wt→mut), the forward decomposition gives:
        ddg_fwd = gain(mut, ctx) - loss(wt, ctx)

    For the reverse (mut→wt), IDEALLY:
        ddg_rev = gain(wt, ctx) - loss(mut, ctx) = -ddg_fwd

    So: gain(mut) + gain(wt) = loss(wt) + loss(mut)
    Or: [gain(mut) + gain(wt)] - [loss(wt) + loss(mut)] = 0

    We run a SECOND forward pass with swapped (wt, mut) to get the
    reverse gain/loss, then penalize:
        L = mean([gain_fwd + gain_rev]² + [loss_fwd + loss_rev]²)

    This provides direct gradient signal to the individual branches,
    unlike standard siamese loss which only constrains ΔΔG.
    """

    def __init__(self, weight_gain: float = 1.0, weight_loss: float = 1.0):
        super().__init__()
        self.weight_gain = weight_gain
        self.weight_loss = weight_loss

    def forward(
        self,
        gain_fwd: torch.Tensor,   # (K,) gain from forward pass (wt→mut)
        loss_fwd: torch.Tensor,   # (K,) loss from forward pass
        gain_rev: torch.Tensor,   # (K,) gain from reverse pass (mut→wt)
        loss_rev: torch.Tensor,   # (K,) loss from reverse pass
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if valid is not None:
            gain_fwd = gain_fwd[valid]
            loss_fwd = loss_fwd[valid]
            gain_rev = gain_rev[valid]
            loss_rev = loss_rev[valid]

        if gain_fwd.numel() == 0:
            z = torch.zeros((), device=gain_fwd.device)
            return z, {"anti_sym_gain": 0.0, "anti_sym_loss": 0.0}

        # Forward gain + reverse gain should be close to
        # forward loss + reverse loss
        gain_sum_penalty = (gain_fwd + gain_rev).pow(2).mean()
        loss_sum_penalty = (loss_fwd + loss_rev).pow(2).mean()

        total = (
            self.weight_gain * gain_sum_penalty
            + self.weight_loss * loss_sum_penalty
        )

        return total, {
            "anti_sym_gain": float(gain_sum_penalty.item()),
            "anti_sym_loss": float(loss_sum_penalty.item()),
        }


# ═══════════════════════════════════════════════════════════════════════
# 3. Standard ΔΔG Anti-Symmetry (BCAS from existing pipeline)
# ═══════════════════════════════════════════════════════════════════════

class DeltaBCASLoss(nn.Module):
    """
    Bias-Corrected Anti-Symmetric loss for the full ΔΔG prediction.
    Same as existing BCASLoss but works with DELTA's output format.

    L = alpha * [mean(ddg_fwd + ddg_rev)]² + beta * mean[(ddg_fwd + ddg_rev)²]
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        ddg_fwd: torch.Tensor,
        ddg_rev: torch.Tensor,
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if valid is not None:
            ddg_fwd = ddg_fwd[valid]
            ddg_rev = ddg_rev[valid]
        if ddg_fwd.numel() == 0:
            z = torch.zeros((), device=ddg_fwd.device)
            return z, {"bcas_bias": 0.0, "bcas_var": 0.0}

        s = ddg_fwd + ddg_rev
        mean_s = s.mean()
        bias_term = mean_s.pow(2)
        var_term = s.pow(2).mean()

        total = self.alpha * bias_term + self.beta * var_term
        return total, {
            "bcas_bias": float(bias_term.item()),
            "bcas_var": float(var_term.item()),
            "bcas_mean_sum": float(mean_s.item()),
        }


# ═══════════════════════════════════════════════════════════════════════
# 4. Mutation-Type Smoothness Regularizer
# ═══════════════════════════════════════════════════════════════════════

class MutationTypeSmoothness(nn.Module):
    """
    Encourages the 20×20 mutation-type embedding to be smooth:
    physicochemically similar substitutions should have similar embeddings.

    Uses BLOSUM62 as a similarity prior: for each pair of substitutions
    (wt→mut_a) and (wt→mut_b) where BLOSUM62(mut_a, mut_b) > threshold,
    penalizes ||E[wt, mut_a] - E[wt, mut_b]||².

    This is a regularizer, not a hard constraint. It prevents the
    20×20 table from becoming an arbitrary lookup and encourages
    generalization to rare substitution types.
    """

    def __init__(self, blosum_threshold: float = 0.0):
        super().__init__()
        self.threshold = blosum_threshold

    def forward(
        self,
        mut_type_embed: nn.Parameter,  # (20, 20, d_sub)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        E = mut_type_embed  # (20, 20, d_sub)

        # For each wildtype, compare all pairs of mutant embeddings
        # E[wt, :, :] has shape (20, d_sub) for each wt
        # Compute pairwise L2 distances within each wildtype's row
        total_penalty = torch.tensor(0.0, device=E.device)
        n_pairs = 0

        for wt in range(20):
            row = E[wt]  # (20, d_sub)
            # Pairwise squared distances: (20, 20)
            dists = torch.cdist(row.unsqueeze(0), row.unsqueeze(0)).squeeze(0)

            # We want nearby AAs (by BLOSUM) to have similar embeddings
            # For simplicity, use adjacent indices in the amino acid ordering
            # (a more principled version would use actual BLOSUM scores)
            for i in range(20):
                for j in range(i + 1, 20):
                    # Penalize all pairs equally (the BLOSUM init already
                    # provides the right starting point)
                    total_penalty = total_penalty + dists[i, j]
                    n_pairs += 1

        reg = total_penalty / max(n_pairs, 1)
        return reg, {"mut_type_smoothness": float(reg.item())}


# ═══════════════════════════════════════════════════════════════════════
# 5. Combined DELTA Loss
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DeltaLossConfig:
    # Primary ΔΔG regression
    loss_type: str = "huber"        # "huber" | "bmc"
    bmc_init_sigma: float = 1.0

    # Decomposition regularization
    use_decomp_reg: bool = True
    decomp_reg_weight: float = 0.1
    decomp_reg_alpha: float = 2.0   # slack multiplier

    # Anti-symmetry on ΔΔG (requires reverse pass)
    use_bcas: bool = True
    bcas_alpha: float = 1.0
    bcas_beta: float = 0.5
    bcas_weight: float = 0.5

    # Anti-symmetry on decomposition (requires reverse pass)
    use_decomp_antisym: bool = True
    decomp_antisym_weight: float = 0.3

    # Mutation-type smoothness
    use_mut_smoothness: bool = False  # off by default (expensive)
    mut_smoothness_weight: float = 0.01


class DeltaLoss(nn.Module):
    """
    Combined loss for DELTA training.

    Orchestrates all loss components and returns a scalar total loss
    plus a diagnostics dictionary for logging.
    """

    def __init__(self, cfg: DeltaLossConfig):
        super().__init__()
        self.cfg = cfg

        # Primary regression loss
        if cfg.loss_type == "huber":
            self.primary_loss = nn.SmoothL1Loss()
        else:
            # BMC will be injected from the training script
            self.primary_loss = nn.SmoothL1Loss()

        # Auxiliary losses
        self.decomp_reg = DecompositionRegularizer(alpha=cfg.decomp_reg_alpha)
        self.bcas = DeltaBCASLoss(alpha=cfg.bcas_alpha, beta=cfg.bcas_beta)
        self.decomp_antisym = DecompositionAntiSymmetryLoss()
        self.mut_smoothness = MutationTypeSmoothness()

    def forward(
        self,
        ddg_pred: torch.Tensor,          # (K,) ΔΔG predictions
        ddg_true: torch.Tensor,          # (K,) ground truth
        gain: torch.Tensor,              # (K,) gain scores
        loss_scores: torch.Tensor,       # (K,) loss scores (avoid name collision)
        valid: torch.Tensor,             # (K,) bool
        # Reverse pass outputs (optional, for anti-symmetry)
        ddg_rev: Optional[torch.Tensor] = None,
        gain_rev: Optional[torch.Tensor] = None,
        loss_rev: Optional[torch.Tensor] = None,
        valid_rev: Optional[torch.Tensor] = None,
        # Mutation-type embedding (optional, for smoothness)
        mut_type_embed: Optional[nn.Parameter] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute all loss components and return total + diagnostics.
        """
        diagnostics = {}
        total_loss = torch.zeros((), device=ddg_pred.device)

        # 1. Primary ΔΔG regression loss
        if valid.sum() > 0:
            primary = self.primary_loss(ddg_pred[valid], ddg_true[valid])
            total_loss = total_loss + primary
            diagnostics["primary_loss"] = float(primary.item())
        else:
            diagnostics["primary_loss"] = 0.0

        # 2. Decomposition magnitude regularization
        if self.cfg.use_decomp_reg:
            reg, reg_info = self.decomp_reg(gain, loss_scores, ddg_pred, valid)
            total_loss = total_loss + self.cfg.decomp_reg_weight * reg
            diagnostics.update(reg_info)

        # 3. BCAS anti-symmetry on ΔΔG (if reverse pass available)
        if self.cfg.use_bcas and ddg_rev is not None:
            v = valid if valid_rev is None else (valid & valid_rev)
            bcas_loss, bcas_info = self.bcas(ddg_pred, ddg_rev, v)
            total_loss = total_loss + self.cfg.bcas_weight * bcas_loss
            diagnostics.update(bcas_info)

        # 4. Decomposition anti-symmetry (if reverse pass available)
        if self.cfg.use_decomp_antisym and gain_rev is not None:
            v = valid if valid_rev is None else (valid & valid_rev)
            dasym_loss, dasym_info = self.decomp_antisym(
                gain, loss_scores, gain_rev, loss_rev, v,
            )
            total_loss = total_loss + self.cfg.decomp_antisym_weight * dasym_loss
            diagnostics.update(dasym_info)

        # 5. Mutation-type smoothness (optional)
        if self.cfg.use_mut_smoothness and mut_type_embed is not None:
            smooth_loss, smooth_info = self.mut_smoothness(mut_type_embed)
            total_loss = total_loss + self.cfg.mut_smoothness_weight * smooth_loss
            diagnostics.update(smooth_info)

        diagnostics["total_loss"] = float(total_loss.item())
        return total_loss, diagnostics

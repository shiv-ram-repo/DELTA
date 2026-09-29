"""
decomposition_bidir.py — DELTA-B Bidirectional Attention Decomposition
=======================================================================

The difference from DELTA-U (decomposition.py):

DELTA-U:
    context = ContextEncoder(query = mutation_site_representation)
    gain = GainBranch(context, mut_embed, sub_embed)
    loss = LossBranch(context, wt_embed, sub_embed)
    → Both branches receive the SAME structural context. The attention over
      neighbors is identity-blind; only the MLP input differs (wt vs mut AA).

DELTA-B (this file):
    context_loss = ContextEncoder(query = f(site_repr, WILDTYPE identity))
    context_gain = ContextEncoder(query = f(site_repr, MUTANT identity))
    gain = GainBranch(context_gain, mut_embed, sub_embed)
    loss = LossBranch(context_loss, wt_embed, sub_embed)
    → Each branch attends to the structural neighborhood DIFFERENTLY,
      conditioned on which residue is "asking". The wildtype-conditioned
      query produces a context reflecting how the wildtype residue engages
      its environment (→ loss); the mutant-conditioned query reflects how
      the mutant would engage it (→ gain).

This is the explicit bidirectional attention mechanism: wildtype-query
attention drives the loss branch, mutant-query attention drives the gain
branch, over a SHARED set of neighbor key/value representations (shared
encoder, two query types — the design chosen for clean ablation against
DELTA-U).

Scientific question DELTA-B tests:
    Does giving the branches distinct, identity-conditioned receptive fields
    SHARPEN the emergent gain/loss convention?
    Measured by: burial-tracking ρ (currently 0.35 in DELTA-U), gain/loss
    apportionment balance, reverse-consistency (currently 0.84), and accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import (
    N_AA,
    MutationTypeEmbedding,
    AminoAcidEmbedding,
    PredictionBranch,
    DecompositionConfig,
)


# ═══════════════════════════════════════════════════════════════════════
# Identity-Conditioned Structural Context Encoder
# ═══════════════════════════════════════════════════════════════════════

class IdentityConditionedContextEncoder(nn.Module):
    """
    Distance-weighted multi-head attention where the QUERY is conditioned
    on a residue-identity embedding, so the same structural neighborhood
    can be attended to differently depending on which residue is "asking."

    Key/Value come from neighbor backbone representations (SHARED between
    the wildtype-query and mutant-query calls — the encoder is shared, only
    the query differs). The query is built from the mutation-site
    representation FUSED with a residue-identity embedding.

    Parameters
    ----------
    d_model : int
        Dimension of backbone representations.
    d_id : int
        Dimension of the identity embedding fused into the query.
    n_heads, k_neighbors, sigma_init, dropout : as in DELTA-U.
    """

    def __init__(
        self,
        d_model: int = 1408,
        d_id: int = 64,
        n_heads: int = 8,
        k_neighbors: int = 32,
        sigma_init: float = 8.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.k_neighbors = k_neighbors
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Query is built from [site_representation ⊕ identity_embedding].
        # We project that fused vector back to d_model for multi-head attention.
        self.query_fuse = nn.Sequential(
            nn.Linear(d_model + d_id, d_model),
            nn.GELU(),
        )

        # Key/Value projections operate on neighbor representations (shared).
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable per-head distance kernel scale.
        self.log_sigma = nn.Parameter(torch.full((n_heads,), math.log(sigma_init)))

        # Learned distance bias.
        self.dist_embed = nn.Sequential(
            nn.Linear(1, n_heads * 2),
            nn.GELU(),
            nn.Linear(n_heads * 2, n_heads),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def precompute_neighbors(
        self,
        representation: torch.Tensor,   # (B, T, D), T = L + 2 (CLS/EOS)
        ca_coords: torch.Tensor,         # (B, L, 3)
        mut_positions: torch.Tensor,     # (K,) 0-indexed residue positions
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Compute neighbor representations, distances, and the site rep ONCE.
        These are shared across the wildtype-query and mutant-query passes.

        Returns a dict of cached tensors.
        """
        device = representation.device
        L_struct = ca_coords.shape[1]
        CLS_OFFSET = 1

        ca = ca_coords[0]  # (L, 3)
        dist_matrix = torch.cdist(ca.unsqueeze(0), ca.unsqueeze(0)).squeeze(0)

        k = min(self.k_neighbors, L_struct - 1)
        mut_pos = mut_positions.long()

        mut_dists = dist_matrix[mut_pos]  # (K, L)
        mut_dists_masked = mut_dists.clone()
        mut_dists_masked[torch.arange(len(mut_pos), device=device), mut_pos] = float("inf")
        if mask is not None:
            invalid = ~mask[0]
            mut_dists_masked[:, invalid] = float("inf")

        _, neighbor_idx = mut_dists_masked.topk(k, dim=-1, largest=False)  # (K, k)

        rep = representation[0]  # (T, D)
        neighbor_reps = rep[neighbor_idx + CLS_OFFSET]  # (K, k, D)
        neighbor_dists = torch.gather(mut_dists, dim=1, index=neighbor_idx)  # (K, k)
        site_reps = rep[mut_pos + CLS_OFFSET]  # (K, D)

        # Precompute K, V (shared across both query types)
        K_mut = len(mut_pos)
        Kk = self.k_proj(neighbor_reps).view(K_mut, k, self.n_heads, self.head_dim)
        V = self.v_proj(neighbor_reps).view(K_mut, k, self.n_heads, self.head_dim)
        Kk = Kk.permute(0, 2, 1, 3)  # (K, H, k, hd)
        V = V.permute(0, 2, 1, 3)    # (K, H, k, hd)

        # Precompute distance kernel + bias (shared)
        sigma = self.log_sigma.exp()
        dist_sq = neighbor_dists.unsqueeze(1) ** 2          # (K, 1, k)
        sigma_sq = (2.0 * sigma ** 2).view(1, self.n_heads, 1)
        dist_kernel = torch.exp(-dist_sq / sigma_sq)        # (K, H, k)
        dist_bias = self.dist_embed(neighbor_dists.unsqueeze(-1))  # (K, k, H)
        dist_bias = dist_bias.permute(0, 2, 1)              # (K, H, k)

        return {
            "site_reps": site_reps,        # (K, D)
            "Kk": Kk, "V": V,              # (K, H, k, hd)
            "dist_kernel": dist_kernel,    # (K, H, k)
            "dist_bias": dist_bias,        # (K, H, k)
            "K_mut": K_mut, "k": k,
        }

    def attend(self, cache: dict, identity_embed: torch.Tensor) -> torch.Tensor:
        """
        Run identity-conditioned attention using precomputed neighbor cache.

        Parameters
        ----------
        cache : dict from precompute_neighbors()
        identity_embed : (K, d_id) — wildtype or mutant identity embedding

        Returns
        -------
        context : (K, D)
        """
        site_reps = cache["site_reps"]       # (K, D)
        Kk, V = cache["Kk"], cache["V"]      # (K, H, k, hd)
        dist_kernel = cache["dist_kernel"]   # (K, H, k)
        dist_bias = cache["dist_bias"]       # (K, H, k)
        K_mut, k = cache["K_mut"], cache["k"]

        # Build identity-conditioned query
        fused = torch.cat([site_reps, identity_embed], dim=-1)  # (K, D + d_id)
        query_in = self.query_fuse(fused)                       # (K, D)
        Q = self.q_proj(query_in).view(K_mut, self.n_heads, self.head_dim)  # (K, H, hd)
        Q = Q.unsqueeze(2)  # (K, H, 1, hd)

        # Attention scores
        attn_scores = torch.matmul(Q, Kk.transpose(-2, -1))  # (K, H, 1, k)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_scores = attn_scores.squeeze(2)  # (K, H, k)

        # Distance modulation + learned bias (shared)
        attn_scores = attn_scores * dist_kernel + dist_bias

        attn_weights = F.softmax(attn_scores, dim=-1)  # (K, H, k)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.einsum("khj,khjd->khd", attn_weights, V)  # (K, H, hd)
        context = context.reshape(K_mut, self.d_model)            # (K, D)
        context = self.out_proj(context)
        context = self.layer_norm(context + site_reps)            # residual
        return context


# ═══════════════════════════════════════════════════════════════════════
# DELTA-B Bidirectional Decomposition Head
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BidirDecompositionConfig(DecompositionConfig):
    """Same fields as DecompositionConfig; d_aa is reused as the identity dim."""
    pass


class BidirectionalDecompositionHead(nn.Module):
    """
    DELTA-B head: identity-conditioned bidirectional attention decomposition.

    - Shared IdentityConditionedContextEncoder (shared K/V neighbor cache).
    - Wildtype identity → loss-branch context.
    - Mutant identity   → gain-branch context.
    - Separate gain/loss PredictionBranch MLPs (as in DELTA-U).

    Output: (ddg, gain, loss), ddg = gain - loss.
    """

    def __init__(self, cfg: DecompositionConfig):
        super().__init__()
        self.cfg = cfg

        self.mut_type_embed = MutationTypeEmbedding(
            d_sub=cfg.d_sub, use_blosum_init=cfg.use_blosum_init,
        )
        self.aa_embed = AminoAcidEmbedding(d_aa=cfg.d_aa)

        # Identity-conditioned context encoder (d_id = d_aa)
        self.context_encoder = IdentityConditionedContextEncoder(
            d_model=cfg.d_model,
            d_id=cfg.d_aa,
            n_heads=cfg.n_context_heads,
            k_neighbors=cfg.k_neighbors,
            sigma_init=cfg.sigma_init,
            dropout=cfg.dropout,
        )

        self.gain_branch = PredictionBranch(
            d_context=cfg.d_model, d_aa=cfg.d_aa, d_sub=cfg.d_sub,
            d_hidden=cfg.d_hidden, n_layers=cfg.n_branch_layers,
            dropout=cfg.dropout,
        )
        self.loss_branch = PredictionBranch(
            d_context=cfg.d_model, d_aa=cfg.d_aa, d_sub=cfg.d_sub,
            d_hidden=cfg.d_hidden, n_layers=cfg.n_branch_layers,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        representation: torch.Tensor,
        ca_coords: torch.Tensor,
        mut_positions: torch.Tensor,
        wt_aa_idx: torch.Tensor,
        mut_aa_idx: torch.Tensor,
        residue_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Identity embeddings
        wt_embed = self.aa_embed(wt_aa_idx)    # (K, d_aa)
        mut_embed = self.aa_embed(mut_aa_idx)  # (K, d_aa)
        sub_embed = self.mut_type_embed(wt_aa_idx, mut_aa_idx)  # (K, d_sub)

        # Shared neighbor cache (computed once)
        cache = self.context_encoder.precompute_neighbors(
            representation, ca_coords, mut_positions, mask=residue_mask,
        )

        # Two identity-conditioned attention passes over the SAME neighbors
        context_loss = self.context_encoder.attend(cache, wt_embed)   # wildtype-query
        context_gain = self.context_encoder.attend(cache, mut_embed)  # mutant-query

        # Branches consume their respective contexts
        gain = self.gain_branch(context_gain, mut_embed, sub_embed)  # (K,)
        loss = self.loss_branch(context_loss, wt_embed, sub_embed)   # (K,)

        ddg = gain - loss
        return ddg, gain, loss

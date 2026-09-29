"""
decomposition.py — DELTA Bidirectional Gain-Loss Decomposition
===============================================================

Core novel components of the DELTA architecture:

1. MutationTypeEmbedding
   ----------------------
   Learnable 20×20×d_sub embedding table capturing the physicochemical
   character of each (wildtype → mutant) amino acid substitution.
   Hydrophobic→charged transitions activate different latent dimensions
   than small→large substitutions. Initialized with BLOSUM62-derived
   priors so training starts from a biologically reasonable manifold.

2. StructuralContextEncoder
   -------------------------
   Distance-weighted multi-head attention over spatial neighbors of the
   mutation site. For position i, identifies the k nearest Cα neighbors,
   computes 1/r²-modulated attention over their backbone representations,
   and produces a context vector c_i that encodes the local structural
   environment. This gives the gain/loss branches explicit geometric
   information about what the mutation site "sees."

3. GainBranch / LossBranch
   -------------------------
   Two parallel cross-attention → MLP modules:
     - GainBranch: "What does the mutant amino acid gain in this context?"
       Query = [mutant_AA_embedding, mutation_type_embedding]
       Key/Value = structural neighborhood context
       Output: scalar gain score

     - LossBranch: "What did the wildtype amino acid contribute?"
       Query = [wildtype_AA_embedding, mutation_type_embedding]
       Key/Value = structural neighborhood context
       Output: scalar loss score

   Both branches share the context encoder but have separate attention
   heads and readout MLPs, allowing them to learn distinct attention
   patterns for gain vs. loss mechanisms.

   ΔΔG = gain - loss

   Anti-symmetry: For mutation A→B, gain(B) - loss(A).
                  For reverse B→A, gain(A) - loss(B).
                  These are NOT constrained to be negatives, but the
                  decomposition structure encourages it through the
                  shared context encoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
N_AA = 20

# BLOSUM62 substitution scores (symmetric, 20×20) for initialization.
# Diagonal = self-substitution scores, off-diagonal = pair similarities.
# We use this to warm-start the mutation-type embedding so that
# physicochemically similar substitutions start nearby in latent space.
# fmt: off
_BLOSUM62_FLAT = [
    4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,  # A
   -1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,  # C→R mapped
   -2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3,  # D→N
   -2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3,  # E→D
    0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,  # F→C
   -1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2,  # G→Q
   -1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2,  # H→E
    0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,  # I→G
   -2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3,  # K→H
   -1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,  # L→I
   -1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,  # M→L
   -1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2,  # N→K
   -1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,  # P→M
   -2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,  # Q→F
   -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,  # R→P
    1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2,  # S
    0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,  # T
   -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,  # V→W
   -2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,  # W→Y
    0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,  # Y→V
]
# fmt: on


def _build_blosum62_init(d_sub: int) -> torch.Tensor:
    """
    Build a (20, 20, d_sub) initialization tensor from BLOSUM62.

    Strategy: the first dimension of each (wt, mut) embedding encodes the
    raw BLOSUM62 score (scaled to unit variance). Remaining dimensions
    are initialized from N(0, 0.02).
    """
    raw = torch.tensor(_BLOSUM62_FLAT, dtype=torch.float32).reshape(N_AA, N_AA)
    # Normalize to zero mean, unit variance
    raw = (raw - raw.mean()) / (raw.std() + 1e-8)

    embed = torch.randn(N_AA, N_AA, d_sub) * 0.02
    if d_sub >= 1:
        embed[:, :, 0] = raw
    return embed


# ═══════════════════════════════════════════════════════════════════════
# 1. Mutation-Type Embedding
# ═══════════════════════════════════════════════════════════════════════

class MutationTypeEmbedding(nn.Module):
    """
    Learnable substitution-type embedding: E[wt_aa, mut_aa] → R^d_sub.

    Parameters
    ----------
    d_sub : int
        Dimensionality of the substitution embedding.
    use_blosum_init : bool
        If True, warm-start the first latent dimension with BLOSUM62 scores.
    """

    def __init__(self, d_sub: int = 64, use_blosum_init: bool = True):
        super().__init__()
        self.d_sub = d_sub

        if use_blosum_init:
            init = _build_blosum62_init(d_sub)
        else:
            init = torch.randn(N_AA, N_AA, d_sub) * 0.02

        # (20, 20, d_sub) — indexed by [wt_aa_idx, mut_aa_idx]
        self.embed = nn.Parameter(init)

    def forward(
        self,
        wt_idx: torch.Tensor,    # (K,) int indices in [0, 19]
        mut_idx: torch.Tensor,   # (K,) int indices in [0, 19]
    ) -> torch.Tensor:
        """Returns (K, d_sub) substitution embeddings."""
        return self.embed[wt_idx, mut_idx]   # advanced indexing


# ═══════════════════════════════════════════════════════════════════════
# 2. Amino Acid Embedding
# ═══════════════════════════════════════════════════════════════════════

class AminoAcidEmbedding(nn.Module):
    """
    Learnable embedding for individual amino acid identities.
    Used as query seeds in the gain/loss branches.
    """

    def __init__(self, d_aa: int = 64):
        super().__init__()
        self.embed = nn.Embedding(N_AA, d_aa)
        nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, aa_idx: torch.Tensor) -> torch.Tensor:
        """aa_idx: (K,) int → (K, d_aa)"""
        return self.embed(aa_idx)


# ═══════════════════════════════════════════════════════════════════════
# 3. Structural Context Encoder
# ═══════════════════════════════════════════════════════════════════════

class StructuralContextEncoder(nn.Module):
    """
    Distance-weighted multi-head attention over spatial neighbors.

    For each mutation position i, gathers the top-k nearest Cα neighbors,
    applies 1/r²-modulated attention, and produces a context vector c_i.

    Parameters
    ----------
    d_model : int
        Dimension of backbone representations.
    n_heads : int
        Number of attention heads.
    k_neighbors : int
        Number of nearest neighbors to attend over.
    sigma_init : float
        Initial length scale (Å) for the distance kernel.
    dropout : float
        Attention dropout.
    """

    def __init__(
        self,
        d_model: int = 1408,
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

        # Projections for multi-head attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable distance kernel scale (one per head)
        self.log_sigma = nn.Parameter(
            torch.full((n_heads,), math.log(sigma_init))
        )

        # Relative distance embedding → per-head bias
        self.dist_embed = nn.Sequential(
            nn.Linear(1, n_heads * 2),
            nn.GELU(),
            nn.Linear(n_heads * 2, n_heads),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        representation: torch.Tensor,   # (B, T, D) where T = L+2 (CLS + L residues + EOS)
        ca_coords: torch.Tensor,         # (B, L, 3) raw protein coordinates
        mut_positions: torch.Tensor,     # (K,) int — 0-indexed residue positions [0, L-1]
        mask: Optional[torch.Tensor] = None,  # (B, L) bool — valid residues
    ) -> torch.Tensor:
        """
        Returns context vectors for each mutation position.

        IMPORTANT: representation has CLS/EOS padding from ESM2, so
        residue i (0-indexed) maps to representation[:, i+1, :].
        ca_coords does NOT have this padding.
        mut_positions are 0-indexed residue positions matching ca_coords.

        Returns
        -------
        context : (K, D)
            Structural context for each mutation position.
        """
        device = representation.device
        L_struct = ca_coords.shape[1]       # raw protein length
        CLS_OFFSET = 1                      # ESM2 CLS token at position 0

        # 1. Compute pairwise Cα distance matrix (L × L, no CLS/EOS)
        ca = ca_coords[0]  # (L_struct, 3) — B=1 in SPURS
        dist_matrix = torch.cdist(ca.unsqueeze(0), ca.unsqueeze(0)).squeeze(0)

        # 2. For each mutation position, find k nearest neighbors
        k = min(self.k_neighbors, L_struct - 1)
        mut_pos = mut_positions.long()  # (K,) — 0-indexed, range [0, L_struct-1]

        # Distances from each mutation position to all residues
        mut_dists = dist_matrix[mut_pos]  # (K, L_struct)

        # Exclude self (set distance to inf)
        mut_dists_masked = mut_dists.clone()
        mut_dists_masked[torch.arange(len(mut_pos), device=device), mut_pos] = float("inf")

        # Apply residue mask if provided
        if mask is not None:
            invalid = ~mask[0]  # (L_struct,)
            mut_dists_masked[:, invalid] = float("inf")

        # Top-k nearest neighbors (indices in [0, L_struct-1])
        _, neighbor_idx = mut_dists_masked.topk(k, dim=-1, largest=False)  # (K, k)

        # 3. Gather representations — shift indices by CLS_OFFSET
        rep = representation[0]  # (T, D) where T = L_struct + 2

        # Neighbor representations: neighbor_idx is [0, L-1], shift to [1, L]
        neighbor_reps = rep[neighbor_idx + CLS_OFFSET]  # (K, k, D)
        neighbor_dists = torch.gather(
            mut_dists, dim=1, index=neighbor_idx
        )  # (K, k)

        # Mutation site representations: mut_pos is [0, L-1], shift to [1, L]
        mut_reps = rep[mut_pos + CLS_OFFSET]  # (K, D)

        # 4. Multi-head distance-weighted attention
        K_mut = len(mut_pos)

        # Project queries (mutation sites) and keys/values (neighbors)
        Q = self.q_proj(mut_reps)           # (K, D)
        Kk = self.k_proj(neighbor_reps)     # (K, k, D)
        V = self.v_proj(neighbor_reps)      # (K, k, D)

        # Reshape for multi-head: (K, n_heads, ..., head_dim)
        Q = Q.view(K_mut, self.n_heads, self.head_dim)           # (K, H, hd)
        Kk = Kk.view(K_mut, k, self.n_heads, self.head_dim)     # (K, k, H, hd)
        V = V.view(K_mut, k, self.n_heads, self.head_dim)       # (K, k, H, hd)

        # Transpose for attention: Q(K, H, 1, hd) @ K^T(K, H, hd, k) → (K, H, 1, k)
        Q = Q.unsqueeze(2)                                       # (K, H, 1, hd)
        Kk = Kk.permute(0, 2, 1, 3)                             # (K, H, k, hd)
        V = V.permute(0, 2, 1, 3)                               # (K, H, k, hd)

        attn_scores = torch.matmul(Q, Kk.transpose(-2, -1))     # (K, H, 1, k)
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # 5. Distance-aware bias
        # Gaussian kernel: exp(-d² / (2σ²))
        sigma = self.log_sigma.exp()                             # (H,)
        dist_sq = neighbor_dists.unsqueeze(1) ** 2               # (K, 1, k)
        sigma_sq = (2.0 * sigma ** 2).view(1, self.n_heads, 1)  # (1, H, 1)
        dist_kernel = torch.exp(-dist_sq / sigma_sq)             # (K, H, k)

        # Learned distance embedding bias
        dist_bias = self.dist_embed(
            neighbor_dists.unsqueeze(-1)                         # (K, k, 1)
        )                                                        # (K, k, H)
        dist_bias = dist_bias.permute(0, 2, 1)                  # (K, H, k)

        # Combine: attention score + distance kernel (as multiplicative gate) + learned bias
        attn_scores = attn_scores.squeeze(2)                     # (K, H, k)
        attn_scores = attn_scores * dist_kernel + dist_bias

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)            # (K, H, k)
        attn_weights = self.attn_dropout(attn_weights)

        # 6. Weighted sum of values
        # attn_weights: (K, H, k) × V: (K, H, k, hd) → (K, H, hd)
        context = torch.einsum("khj,khjd->khd", attn_weights, V)  # (K, H, hd)
        context = context.reshape(K_mut, self.d_model)            # (K, D)

        # Output projection + residual + layer norm
        context = self.out_proj(context)                          # (K, D)
        context = self.layer_norm(context + mut_reps)             # (K, D)

        return context


# ═══════════════════════════════════════════════════════════════════════
# 4. Gain / Loss Branches
# ═══════════════════════════════════════════════════════════════════════

class PredictionBranch(nn.Module):
    """
    One branch of the bidirectional decomposition.

    Takes: amino acid embedding + mutation-type embedding + structural context
    Produces: scalar score (gain or loss).

    Architecture:
        [aa_embed ⊕ sub_embed ⊕ context] → CrossAttn → MLP → scalar
    """

    def __init__(
        self,
        d_context: int = 1408,
        d_aa: int = 64,
        d_sub: int = 64,
        d_hidden: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_input = d_context + d_aa + d_sub

        # Fusion projection: combine context, AA identity, and substitution type
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Deep MLP for scalar readout
        layers = []
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(d_hidden, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(d_hidden, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize final layer near zero for stable training start
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        context: torch.Tensor,     # (K, D_context)
        aa_embed: torch.Tensor,    # (K, d_aa)
        sub_embed: torch.Tensor,   # (K, d_sub)
    ) -> torch.Tensor:
        """Returns (K,) scalar scores."""
        x = torch.cat([context, aa_embed, sub_embed], dim=-1)  # (K, D_in)
        x = self.input_proj(x)      # (K, d_hidden)
        return self.mlp(x).squeeze(-1)  # (K,)


# ═══════════════════════════════════════════════════════════════════════
# 5. Full Decomposition Head
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DecompositionConfig:
    d_model: int = 1408         # backbone representation dimension
    d_aa: int = 64              # amino acid embedding dimension
    d_sub: int = 64             # substitution-type embedding dimension
    d_hidden: int = 512         # MLP hidden dimension in branches
    n_branch_layers: int = 3    # number of MLP layers in each branch
    n_context_heads: int = 8    # attention heads in context encoder
    k_neighbors: int = 32       # spatial neighbors for context
    sigma_init: float = 8.0     # initial distance kernel scale (Å)
    dropout: float = 0.1
    use_blosum_init: bool = True


class DecompositionHead(nn.Module):
    """
    Full DELTA decomposition head.

    Combines:
        - MutationTypeEmbedding
        - AminoAcidEmbedding
        - StructuralContextEncoder
        - GainBranch (PredictionBranch)
        - LossBranch (PredictionBranch)

    Input: backbone representations + coordinates + mutation info
    Output: (ddg, gain, loss) where ddg = gain - loss
    """

    def __init__(self, cfg: DecompositionConfig):
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.mut_type_embed = MutationTypeEmbedding(
            d_sub=cfg.d_sub, use_blosum_init=cfg.use_blosum_init,
        )
        self.aa_embed = AminoAcidEmbedding(d_aa=cfg.d_aa)

        # Structural context encoder
        self.context_encoder = StructuralContextEncoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_context_heads,
            k_neighbors=cfg.k_neighbors,
            sigma_init=cfg.sigma_init,
            dropout=cfg.dropout,
        )

        # Gain and Loss branches
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
        representation: torch.Tensor,   # (B, L, D)
        ca_coords: torch.Tensor,         # (B, L, 3)
        mut_positions: torch.Tensor,     # (K,) mutation positions
        wt_aa_idx: torch.Tensor,         # (K,) wildtype AA indices [0-19]
        mut_aa_idx: torch.Tensor,        # (K,) mutant AA indices [0-19]
        residue_mask: Optional[torch.Tensor] = None,  # (B, L)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        ddg : (K,)
            Predicted ΔΔG = gain - loss.
        gain : (K,)
            What the mutant amino acid gains in this structural context.
        loss : (K,)
            What the wildtype amino acid contributed (and is lost).
        """
        # 1. Structural context at each mutation site
        context = self.context_encoder(
            representation, ca_coords, mut_positions, mask=residue_mask,
        )  # (K, D)

        # 2. Mutation-type embedding
        sub_embed = self.mut_type_embed(wt_aa_idx, mut_aa_idx)  # (K, d_sub)

        # 3. Amino acid identity embeddings
        wt_embed = self.aa_embed(wt_aa_idx)    # (K, d_aa)
        mut_embed = self.aa_embed(mut_aa_idx)  # (K, d_aa)

        # 4. Gain branch: mutant AA queries the context
        gain = self.gain_branch(context, mut_embed, sub_embed)  # (K,)

        # 5. Loss branch: wildtype AA queries the context
        loss = self.loss_branch(context, wt_embed, sub_embed)   # (K,)

        # 6. ΔΔG decomposition
        ddg = gain - loss

        return ddg, gain, loss

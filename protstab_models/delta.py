"""
delta.py — DELTA: Decomposed Estimation of Local Thermodynamic Attributes
==========================================================================

Full model architecture for protein stability prediction via bidirectional
gain-loss decomposition with mutation-type conditioning.

Architecture overview:

    ┌─────────────────────────────────────────┐
    │  Backbone (shared with existing repo)   │
    │  ProteinMPNN encoder → structural feats │
    │  ESM2 + adapter → sequence feats        │
    │  Cross-attention fusion                 │
    │  Concatenation → (B, L, 1408)           │
    └──────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────┐
    │  DELTA Decomposition Head (NEW)         │
    │                                         │
    │  For each mutation at position i:       │
    │  ┌───────────────────────────────────┐  │
    │  │ StructuralContextEncoder          │  │
    │  │ k-nearest Cα neighbor attention   │  │
    │  │ Distance-weighted pooling → c_i   │  │
    │  └──────────────┬────────────────────┘  │
    │                 │                       │
    │  ┌──────────────┴────────────────────┐  │
    │  │ MutationTypeEmbedding             │  │
    │  │ E[wt_aa, mut_aa] → e_sub          │  │
    │  └──────────────┬────────────────────┘  │
    │                 │                       │
    │     ┌───────────┴──────────┐            │
    │     │                      │            │
    │  ┌──▼─────────┐   ┌───────▼────────┐   │
    │  │ GainBranch  │   │  LossBranch    │   │
    │  │ mut_aa→attn │   │  wt_aa→attn    │   │
    │  │  → scalar   │   │   → scalar     │   │
    │  └──┬──────────┘   └───────┬────────┘   │
    │     │                      │            │
    │     └──────┐   ┌───────────┘            │
    │            ▼   ▼                        │
    │     ΔΔG = gain - loss                   │
    └─────────────────────────────────────────┘

Key differences from the original SPURS model:
  1. Replaces the 21-class energy MLP head with decomposition branches
  2. Adds explicit structural context via distance-weighted neighbor attention
  3. Adds mutation-type conditioning via BLOSUM62-initialized embeddings
  4. Predicts ΔΔG as gain − loss instead of energy(mut) − energy(wt)
  5. Exposes gain/loss separately for interpretability and auxiliary losses

Drop-in replacement: uses the same data pipeline, batch format, and
evaluation infrastructure. The only change in the training loop is the
loss function (delta_loss.py provides the decomposition-aware losses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .mlp import MLP, MLPConfig
from .protein_mpnn import ProteinMPNN, ProteinMPNNConfig, get_protein_mpnn
from .esm_adapter import ESM2WithStructuralAdapter
from .decomposition import DecompositionHead, DecompositionConfig
from .decomposition_bidir import BidirectionalDecompositionHead


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DELTAConfig:
    # Backbone config (inherited from SPURS)
    encoder: ProteinMPNNConfig = field(default_factory=ProteinMPNNConfig)
    adapter_layer_indices: List = field(default_factory=lambda: [-1])
    separate_loss: bool = True
    name: str = "esm2_t33_650M_UR50D"
    dropout: float = 0.1
    mlp: MLPConfig = field(default_factory=MLPConfig)
    proteinmpnn_ckpt: str = ""

    # Decomposition head config
    decomposition: DecompositionConfig = field(
        default_factory=DecompositionConfig
    )

    # DELTA-B: use identity-conditioned bidirectional attention decomposition.
    # False → DELTA-U (shared context, two heads).
    # True  → DELTA-B (wildtype-query → loss, mutant-query → gain).
    bidirectional: bool = False

    # Whether to keep the original MLP head as an auxiliary predictor
    # (for ablation: compare decomposition vs. energy-table prediction)
    use_auxiliary_energy_head: bool = False


# ──────────────────────────────────────────────────────────────────────
# Helper: extract AA indices from one-hot append_tensors
# ──────────────────────────────────────────────────────────────────────

def _onehot_to_idx(onehot: torch.Tensor) -> torch.Tensor:
    """
    Convert (K, 21) or (K, 20) one-hot to (K,) integer indices.
    If 21-dim, the last class (index 20) = unknown/gap → clamp to 0.
    Maps to our 20-letter ALPHABET ordering.
    """
    idx = onehot.argmax(dim=-1)  # (K,)
    # Clamp any index >= 20 (unknown) to 0 (will produce garbage predictions
    # but won't crash; these should be filtered by the valid mask anyway)
    idx = idx.clamp(max=19)
    return idx


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

class DELTA(BaseModel):
    """
    DELTA: Decomposed Estimation of Local Thermodynamic Attributes.

    Protein stability prediction through bidirectional gain-loss
    decomposition with mutation-type conditioning.
    """

    _default_cfg = DELTAConfig()

    def __init__(self, cfg: DELTAConfig) -> None:
        super().__init__(cfg)

        # ── Extract config values ─────────────────────────────────
        encoder_cfg = self.cfg["encoder"] if isinstance(self.cfg, dict) else self.cfg.encoder
        mlp_cfg = self.cfg["mlp"] if isinstance(self.cfg, dict) else self.cfg.mlp
        name = self.cfg["name"] if isinstance(self.cfg, dict) else self.cfg.name
        dropout = self.cfg["dropout"] if isinstance(self.cfg, dict) else self.cfg.dropout
        adapter_idx = (
            self.cfg["adapter_layer_indices"] if isinstance(self.cfg, dict)
            else self.cfg.adapter_layer_indices
        )
        ckpt_path = (
            self.cfg["proteinmpnn_ckpt"] if isinstance(self.cfg, dict)
            else getattr(self.cfg, "proteinmpnn_ckpt", "")
        )
        decomp_cfg = (
            self.cfg["decomposition"] if isinstance(self.cfg, dict)
            else self.cfg.decomposition
        )
        use_aux = (
            self.cfg["use_auxiliary_energy_head"] if isinstance(self.cfg, dict)
            else getattr(self.cfg, "use_auxiliary_energy_head", False)
        )

        def _g(c, k):
            return c[k] if isinstance(c, dict) else getattr(c, k)

        self.tune = _g(encoder_cfg, "tune")
        self.use_input_decoding_order = _g(encoder_cfg, "use_input_decoding_order")

        # ── 1. Encoder: ProteinMPNN ──────────────────────────────
        self.encoder = get_protein_mpnn(
            tune=self.tune,
            ckpt_path=ckpt_path if ckpt_path else None,
        )

        if isinstance(encoder_cfg, dict):
            encoder_cfg["d_model"] = _g(mlp_cfg, "input_dim")
        else:
            encoder_cfg.d_model = _g(mlp_cfg, "input_dim")

        # ── 2. Decoder: ESM2 + structural adapter ────────────────
        from dataclasses import dataclass as _dc, field as _f

        @_dc
        class _AdapterArgs:
            adapter_layer_indices: list = _f(default_factory=list)
            dropout: float = 0.1
            encoder: object = None

        adapter_args = _AdapterArgs(
            adapter_layer_indices=list(adapter_idx),
            dropout=dropout,
            encoder=encoder_cfg,
        )
        self.decoder = ESM2WithStructuralAdapter.from_pretrained(
            args=adapter_args, name=name,
        )

        # ── 3. Cross-Attention Fusion ────────────────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=1280,
            kdim=128,
            vdim=128,
            num_heads=8,
            batch_first=True,
        )

        # ── 4. Optional flat projection ──────────────────────────
        flat_dim = _g(mlp_cfg, "flat_dim")
        if flat_dim > 0:
            self.flat_layers = nn.Linear(1280, flat_dim)
            self.dp = nn.Dropout(dropout)
        else:
            self.flat_layers = None
            self.dp = None

        self.input_dim = _g(encoder_cfg, "d_model")

        # ── 5. Compute backbone output dimension ─────────────────
        if flat_dim < 0:
            backbone_dim = self.input_dim + 1280
        elif flat_dim > 0:
            backbone_dim = self.input_dim + flat_dim
        else:
            backbone_dim = self.input_dim + 1280

        # ── 6. DELTA Decomposition Head ──────────────────────────
        # Read the bidirectional flag (DELTA-B vs DELTA-U)
        use_bidir = (
            self.cfg["bidirectional"] if isinstance(self.cfg, dict)
            else getattr(self.cfg, "bidirectional", False)
        )
        self.bidirectional = use_bidir
        HeadClass = BidirectionalDecompositionHead if use_bidir else DecompositionHead

        if isinstance(decomp_cfg, dict):
            decomp_cfg["d_model"] = backbone_dim
            self.decomp_head = HeadClass(
                DecompositionConfig(**decomp_cfg)
            )
        else:
            decomp_cfg.d_model = backbone_dim
            self.decomp_head = HeadClass(decomp_cfg)

        # ── 7. Optional auxiliary energy head (for ablation) ─────
        self.use_aux = use_aux
        if use_aux:
            aux_mlp_cfg = MLPConfig()
            aux_mlp_cfg.input_dim = backbone_dim + 42  # + append_tensors
            aux_mlp_cfg.hidden_dim = [512, 512, 512]
            aux_mlp_cfg.output_dim = 21
            aux_mlp_cfg.dropout = 0.1
            self.aux_mlp = MLP(aux_mlp_cfg)
        else:
            self.aux_mlp = None

        # Expose tokeniser metadata
        self.padding_idx = self.decoder.padding_idx
        self.mask_idx = self.decoder.mask_idx
        self.cls_idx = self.decoder.cls_idx
        self.eos_idx = self.decoder.eos_idx

    # ──────────────────────────────────────────────────────────────
    # Backbone forward (shared with SPURS)
    # ──────────────────────────────────────────────────────────────

    def forward_encoder(self, batch: dict) -> torch.Tensor:
        """Run ProteinMPNN encoder → concatenated structural features."""
        X = batch["X"]
        S = batch["S"]
        mask = batch["mask"]
        chain_M = batch["chain_M"]
        residue_idx = batch["residue_idx"]
        chain_encoding_all = batch["chain_encoding_all"]

        all_mpnn_hid, mpnn_embed, _ = self.encoder(
            X, S, mask, chain_M, residue_idx, chain_encoding_all,
            None, self.use_input_decoding_order,
        )
        return torch.cat(
            [all_mpnn_hid[0], mpnn_embed, all_mpnn_hid[1], all_mpnn_hid[2]],
            dim=-1,
        )

    def forward_backbone(self, batch: dict) -> torch.Tensor:
        """
        Full backbone: ProteinMPNN → ESM2+adapter → cross-attention → concat.
        Returns (B, T, D_backbone) representation.
        """
        # 1. Encoder
        if not self.tune:
            with torch.no_grad():
                batch["feats"] = self.forward_encoder(batch)
        else:
            batch["feats"] = self.forward_encoder(batch)

        # 2. Truncate encoder features
        batch["feats"] = batch["feats"][:, :, :self.input_dim]
        encoder_out = {"feats": F.pad(batch["feats"], (0, 0, 1, 1))}

        # 3. ESM2 + adapter
        decoder_out = self.decoder(
            tokens=batch["tokens"],
            encoder_out=encoder_out,
        )
        representation = decoder_out["representations"][-1]

        # 4. Optional flat projection
        if self.flat_layers is not None:
            representation = self.flat_layers(representation)
            representation = self.dp(representation)
            representation = F.gelu(representation)

        # 5. Cross-attention fusion
        mpnn_feats = encoder_out["feats"]
        attn_out, _ = self.cross_attn(
            query=representation,
            key=mpnn_feats,
            value=mpnn_feats,
        )
        representation = representation + attn_out

        # 6. Concatenate
        representation = torch.cat([representation, mpnn_feats], dim=-1)

        return representation, mpnn_feats

    # ──────────────────────────────────────────────────────────────
    # DELTA forward
    # ──────────────────────────────────────────────────────────────

    def forward(
        self,
        batch: dict,
        return_decomposition: bool = False,
        **kwargs,
    ):
        """
        Forward pass.

        Parameters
        ----------
        batch : dict
            Standard batch from the data pipeline.
        return_decomposition : bool
            If True, returns (ddg, gain, loss) tuple instead of just ddg.

        Returns
        -------
        ddg : (K,) predicted ΔΔG values.
        Or (ddg, gain, loss) if return_decomposition=True.
        """
        # 1. Run backbone
        representation, mpnn_feats = self.forward_backbone(batch)

        # 2. Extract mutation info
        mut_ids = (
            batch["mut_ids"] if isinstance(batch["mut_ids"], torch.Tensor)
            else torch.tensor(batch["mut_ids"])
        ).to(representation.device)

        # 3. Extract wildtype and mutant AA indices from append_tensors
        append_tensors = batch["append_tensors"]  # (K, 42)
        wt_onehot = append_tensors[:, :21].float()
        mut_onehot = append_tensors[:, 21:].float()

        wt_aa_idx = _onehot_to_idx(wt_onehot)     # (K,)
        mut_aa_idx = _onehot_to_idx(mut_onehot)   # (K,)

        # 4. Extract Cα coordinates
        # batch["X"] has shape (B, L, 4, 3) where dim 1 = [N, CA, C, O]
        ca_coords = batch["X"][:, :, 1, :]  # (B, L, 3)

        # 5. Run DELTA decomposition head
        # mut_ids are 0-indexed residue positions [0, L-1].
        # ca_coords is (B, L, 3) — same indexing as mut_ids.
        # representation is (B, L+2, D) due to ESM's CLS/EOS tokens.
        # The decomposition head handles the +1 CLS offset internally
        # when indexing into representation, so we pass raw mut_ids here.

        ddg, gain, loss = self.decomp_head(
            representation=representation,
            ca_coords=ca_coords,
            mut_positions=mut_ids.long(),
            wt_aa_idx=wt_aa_idx.to(representation.device),
            mut_aa_idx=mut_aa_idx.to(representation.device),
        )

        # Handle NaN safety
        ddg[torch.isnan(ddg)] = 10000.0

        # Store intermediate representations for potential auxiliary losses
        batch["_delta_gain"] = gain
        batch["_delta_loss"] = loss
        batch["_delta_context"] = None  # populated by context encoder

        if return_decomposition:
            return ddg, gain, loss

        return ddg

    def forward_all_mutations(self, batch: dict) -> torch.Tensor:
        """
        Predict ΔΔG for ALL 20 possible mutations at every position.

        Returns (L, 20) matrix of ΔΔG predictions.
        Used for full-protein saturation mutagenesis scanning.
        """
        # Run backbone
        representation, _ = self.forward_backbone(batch)
        ca_coords = batch["X"][:, :, 1, :]

        wt_seq = batch["seq"]
        L = len(wt_seq)
        device = representation.device

        all_ddg = torch.zeros(L, N_AA, device=device)

        for pos in range(L):
            wt_aa = ALPHABET.index(wt_seq[pos])
            shifted_pos = torch.tensor([pos + 1], device=device)

            for mut_aa in range(N_AA):
                if mut_aa == wt_aa:
                    continue  # self-mutation = 0

                wt_idx = torch.tensor([wt_aa], device=device)
                mut_idx = torch.tensor([mut_aa], device=device)

                ddg, _, _ = self.decomp_head(
                    representation=representation,
                    ca_coords=ca_coords,
                    mut_positions=shifted_pos,
                    wt_aa_idx=wt_idx,
                    mut_aa_idx=mut_idx,
                )
                all_ddg[pos, mut_aa] = ddg.item()

        return all_ddg


# ──────────────────────────────────────────────────────────────────────
# Convenience: number of amino acids
# ──────────────────────────────────────────────────────────────────────
N_AA = 20

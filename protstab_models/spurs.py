"""
SPURS with Cross-Attention Fusion (Option 2)
============================================
Full SPURS model: ProteinMPNN encoder + ESM2-with-structural-adapter decoder
+ Cross-Attention Fusion layer + MLP head.

Fuses sequence representations and 3D structural representations through an
explicit cross-attention bottleneck right before the final MLP head, forcing
the network to reconcile sequence features with geometric structural constraints.

ΔΔG prediction (all-mutants-per-pass):
    1. ProteinMPNN encodes wild-type structure → structural features
    2. ESM2 + adapter combines sequence + structure → per-residue representation
    3. Cross-Attention: ESM acts as Query; ProteinMPNN acts as Key and Value
    4. Representation is concatenated with raw encoder feats
    5. MLP head produces (B, K, 21) per-AA energies for each mutated position
    6. ΔΔG = g(mut) - g(wt) via one-hot masking from `append_tensors`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base         import BaseModel
from .mlp          import MLP, MLPConfig
from .protein_mpnn import ProteinMPNN, ProteinMPNNConfig, get_protein_mpnn
from .esm_adapter  import ESM2WithStructuralAdapter


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SPURSConfig:
    encoder: ProteinMPNNConfig    = field(default_factory=ProteinMPNNConfig)
    adapter_layer_indices: List   = field(default_factory=lambda: [-1])
    separate_loss: bool           = True
    name: str                     = "esm2_t33_650M_UR50D"
    dropout: float                = 0.1
    mlp: MLPConfig                = field(default_factory=MLPConfig)

    # Optional: path to ProteinMPNN v_48_020 checkpoint
    proteinmpnn_ckpt: str         = ""


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class SPURS(BaseModel):
    """Reproduction of SPURS augmented with a late Cross-Attention Bottleneck."""

    _default_cfg = SPURSConfig()

    def __init__(self, cfg: SPURSConfig) -> None:
        super().__init__(cfg)

        # ── Pull values from cfg (handle dict or omegaconf) ────────────
        encoder_cfg = self.cfg["encoder"]            if isinstance(self.cfg, dict) else self.cfg.encoder
        mlp_cfg     = self.cfg["mlp"]                if isinstance(self.cfg, dict) else self.cfg.mlp
        name        = self.cfg["name"]               if isinstance(self.cfg, dict) else self.cfg.name
        dropout     = self.cfg["dropout"]            if isinstance(self.cfg, dict) else self.cfg.dropout
        adapter_idx = (
            self.cfg["adapter_layer_indices"] if isinstance(self.cfg, dict)
            else self.cfg.adapter_layer_indices
        )
        ckpt_path   = (
            self.cfg["proteinmpnn_ckpt"] if isinstance(self.cfg, dict)
            else getattr(self.cfg, "proteinmpnn_ckpt", "")
        )

        def _g(c, k):
            return c[k] if isinstance(c, dict) else getattr(c, k)

        self.tune                     = _g(encoder_cfg, "tune")
        self.use_input_decoding_order = _g(encoder_cfg, "use_input_decoding_order")

        # ── 1. Encoder: ProteinMPNN ────────────────────────────────────
        self.encoder = get_protein_mpnn(
            tune=self.tune,
            ckpt_path=ckpt_path if ckpt_path else None,
        )

        # Patch encoder cfg's d_model to match MLP input expectation
        if isinstance(encoder_cfg, dict):
            encoder_cfg["d_model"] = _g(mlp_cfg, "input_dim")
        else:
            encoder_cfg.d_model = _g(mlp_cfg, "input_dim")

        # ── 2. Decoder: ESM2 + structural adapter ──────────────────────
        from dataclasses import dataclass as _dc, field as _f

        @_dc
        class _AdapterArgs:
            adapter_layer_indices: list = _f(default_factory=list)
            dropout:               float = 0.1
            encoder:               object = None

        adapter_args = _AdapterArgs(
            adapter_layer_indices=list(adapter_idx),
            dropout=dropout,
            encoder=encoder_cfg,
        )
        self.decoder = ESM2WithStructuralAdapter.from_pretrained(
            args=adapter_args, name=name,
        )

        # ── 3. MLP head ────────────────────────────────────────────────
        flat_dim = _g(mlp_cfg, "flat_dim")
        if flat_dim < 0:
            new_input_dim = _g(mlp_cfg, "input_dim") + 1280
        else:
            new_input_dim = _g(mlp_cfg, "input_dim") + flat_dim

        if isinstance(mlp_cfg, dict):
            mlp_cfg["input_dim"] = new_input_dim
        else:
            mlp_cfg.input_dim = new_input_dim

        self.input_dim = _g(encoder_cfg, "d_model")   # = original mlp.input_dim
        self.mlp       = MLP(mlp_cfg)

        # Optional flat projection of ESM features
        if flat_dim > 0:
            self.flat_layers = nn.Linear(1280, flat_dim)
            self.dp          = nn.Dropout(dropout)
        else:
            self.flat_layers = None
            self.dp          = None

        # =====================================================================
        # OPTION 2: CROSS-ATTENTION BOTTLENECK LAYER
        # =====================================================================
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=1280,  # ESM embedding dimension (Query)
            kdim=128,        # ProteinMPNN feature dimension (Key)
            vdim=128,        # ProteinMPNN feature dimension (Value)
            num_heads=8,
            batch_first=True
        )
        # =====================================================================

        # Expose tokeniser metadata
        self.padding_idx = self.decoder.padding_idx
        self.mask_idx    = self.decoder.mask_idx
        self.cls_idx     = self.decoder.cls_idx
        self.eos_idx     = self.decoder.eos_idx

    def forward_encoder(self, batch: dict) -> torch.Tensor:
        X                  = batch["X"]
        S                  = batch["S"]
        mask               = batch["mask"]
        chain_M            = batch["chain_M"]
        residue_idx        = batch["residue_idx"]
        chain_encoding_all = batch["chain_encoding_all"]

        all_mpnn_hid, mpnn_embed, _ = self.encoder(
            X, S, mask, chain_M, residue_idx, chain_encoding_all,
            None, self.use_input_decoding_order,
        )
        return torch.cat(
            [all_mpnn_hid[0], mpnn_embed, all_mpnn_hid[1], all_mpnn_hid[2]],
            dim=-1,
        )

    def forward(self, batch: dict, **kwargs):
        # 1. Run encoder (optionally with frozen weights)
        if not self.tune:
            with torch.no_grad():
                batch["feats"] = self.forward_encoder(batch)
        else:
            batch["feats"] = self.forward_encoder(batch)

        # 2. Truncate encoder features to the size the adapter expects
        batch["feats"] = batch["feats"][:, :, : self.input_dim]
        encoder_out    = {"feats": F.pad(batch["feats"], (0, 0, 1, 1))}

        # 3. Run ESM2 + adapter
        decoder_out    = self.decoder(
            tokens=batch["tokens"],
            encoder_out=encoder_out,
        )
        representation = decoder_out["representations"][-1]   # (B, T, embed_dim)

        # 4. Optional flat projection of ESM features
        if self.flat_layers is not None:
            representation = self.flat_layers(representation)
            representation = self.dp(representation)
            representation = F.gelu(representation)

        # =====================================================================
        # OPTION 2: CROSS-ATTENTION FUSION
        # Intercept and route tensors before naive late-stage concatenation
        # =====================================================================
        mpnn_feats = encoder_out["feats"]
        attn_out, _ = self.cross_attn(
            query=representation,
            key=mpnn_feats,
            value=mpnn_feats
        )
        # Residual connection combining linguistic and forced structural context
        representation = representation + attn_out
        # =====================================================================

        # 5. Concat with encoder features (keeps dimensions completely identical)
        representation = torch.cat([representation, mpnn_feats], dim=-1)

        # ── Mode A: return full (L, 20) ΔΔG table for all mutations ──
        if kwargs.get("return_logist", False):
            wt              = batch["seq"]
            L               = len(wt)
            shifted_mut_ids = torch.repeat_interleave(torch.arange(1, 1 + L), 20)
            muted_id_repr   = representation[:, shifted_mut_ids.long()]
            batch["muted_id_representation"] = muted_id_repr

            pre_output = self.mlp(batch)         # (B, L*20, 21)
            ddg_out    = pre_output.squeeze(0)   # (L*20, 21)

            mt_aa = torch.arange(20).repeat(L)
            ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
            wt_aa = torch.repeat_interleave(
                torch.tensor([ALPHABET.index(s) for s in wt]), 20,
            )
            num_classes = 21
            mt_oh = F.one_hot(mt_aa, num_classes=num_classes).to(representation.device)
            wt_oh = F.one_hot(wt_aa, num_classes=num_classes).to(representation.device)

            ddg_mut = (ddg_out * mt_oh).sum(-1)
            ddg_wt  = (ddg_out * wt_oh).sum(-1)
            return (ddg_mut - ddg_wt).reshape(-1, 20)

        # ── Mode B: scalar ΔΔG per supplied mutation ──
        mut_ids = (
            batch["mut_ids"] if isinstance(batch["mut_ids"], torch.Tensor)
            else torch.tensor(batch["mut_ids"])
        )
        shifted_ids = mut_ids.to(representation.device) + 1
        muted_id_repr = representation[:, shifted_ids.long()]
        batch["muted_id_representation"] = muted_id_repr

        pre_output = self.mlp(batch)             # (B, K, 21)
        ddg_out    = pre_output.squeeze()        # (K, 21) when B=1

        wt_onehot  = batch["append_tensors"][:, :21]
        mut_onehot = batch["append_tensors"][:, 21:]
        ddg_mut    = (ddg_out * mut_onehot).sum(-1)
        ddg_wt     = (ddg_out * wt_onehot).sum(-1)
        ddg        = ddg_mut - ddg_wt
        ddg[torch.isnan(ddg)] = 10000.0
        return ddg

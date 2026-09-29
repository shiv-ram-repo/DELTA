"""
mlp
===
Final stability prediction head.

Mirrors SPURS's `MLP` but with `output_dim=21` as the correct default (SPURS's
class default of 1 is wrong for the all-mutants-per-pass scheme and is
overridden at runtime by Hydra).

Input  : the per-mutation representation vector concatenated from ESM2 (or its
         flat projection) and the ProteinMPNN structural features.
         Shape: (B, K, input_dim) where K = number of mutations in the batch.
Output : (B, K, 21) — one (negative) free-energy value per amino acid identity.

ΔΔG for mutation `wt -> mut` at position i is then computed in the parent
SPURS model as:
    g_mut = (mlp_out * mut_onehot_21).sum(-1)
    g_wt  = (mlp_out * wt_onehot_21).sum(-1)
    ddG   = g_mut - g_wt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


@dataclass
class MLPConfig:
    num_layers: int   = 3
    input_dim:  int   = 2560 + 42      # default SPURS sizing; will be set explicitly
    hidden_dim: List[int] = field(
    default_factory=lambda: [512, 512, 512]
    )
    output_dim: int   = 21              # MUST be 21 for SPURS-style head
    dropout:    float = 0.1
    ckpt_path:  str   = ""
    append_tensors: bool = True
    flat_dim:   int   = -1              # if > 0, ESM features are pre-flattened
                                        # to this dim before concat with encoder feats


class MLP(BaseModel):
    """
    Stacks `n-1` GELU + dropout linear layers followed by a final linear
    projection to `output_dim`.
    """

    _default_cfg = MLPConfig()

    def __init__(self, cfg: MLPConfig) -> None:
        super().__init__(cfg)

        # 1. Extract raw config values
        input_dim   = self.cfg["input_dim"]   if isinstance(self.cfg, dict) else self.cfg.input_dim
        hidden_cfg  = self.cfg["hidden_dim"]  if isinstance(self.cfg, dict) else self.cfg.hidden_dim
        output_dim  = self.cfg["output_dim"]  if isinstance(self.cfg, dict) else self.cfg.output_dim
        dropout     = self.cfg["dropout"]     if isinstance(self.cfg, dict) else self.cfg.dropout
        ckpt_path   = self.cfg["ckpt_path"]   if isinstance(self.cfg, dict) else self.cfg.ckpt_path
        self.append_tensors = (
            self.cfg["append_tensors"] if isinstance(self.cfg, dict) else self.cfg.append_tensors
        )

        # 2. FIX: Coerce input/output dims to explicit integers
        input_dim = int(input_dim)
        output_dim = int(output_dim)

        # 3. FIX: Safely handle Hydra ListConfigs or Tuples by checking for iterability
        if hasattr(hidden_cfg, "__iter__") and not isinstance(hidden_cfg, str):
            hidden_cfg = list(hidden_cfg)

        # Always 3 hidden layers when hidden_dim is a scalar
        hidden_dim = hidden_cfg if isinstance(hidden_cfg, list) else [hidden_cfg] * 3

        # 4. FIX: Ensure all dimensions inside the list are explicitly integers
        hidden_dim = [int(dim) for dim in hidden_dim]

        num_layers = len(hidden_dim) + 1

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim[0], bias=True)]
            + [
                nn.Linear(hidden_dim[i], hidden_dim[i + 1], bias=True)
                for i in range(num_layers - 2)
            ]
            + [nn.Linear(hidden_dim[-1], output_dim, bias=True)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(num_layers - 1)]
        )

        self._initialise_weights(ckpt_path)

    def _initialise_weights(self, ckpt_path: str = ""):
        if not ckpt_path:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            sd = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(sd)

    def forward(self, batch, return_embed: bool = False):
        """
        Expects `batch` to be a dict with either:
            - 'muted_id_representation' : (B, K, input_dim)  — preferred
            - 'mpnn_outputs'            : (B, K, input_dim)  — legacy fallback
        """
        x = batch.get("muted_id_representation", batch.get("mpnn_outputs", None))
        for i in range(len(self.fcs) - 1):
            x = self.fcs[i](x)
            x = self.dropouts[i](x)
            x = F.gelu(x)
        if return_embed:
            return x
        return self.fcs[-1](x)

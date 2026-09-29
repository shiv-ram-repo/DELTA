# DELTA: Decomposed Estimation of Local Thermodynamic Attributes

Protein stability prediction through bidirectional gain-loss decomposition
with mutation-type conditioning.

## Key Innovation

DELTA decomposes the ΔΔG prediction into two interpretable components:

- **Gain**: What the mutant amino acid gains from the structural context
- **Loss**: What the wildtype amino acid contributed (and is lost upon mutation)
- **ΔΔG = Gain − Loss**

This decomposition is conditioned on a learned 20×20 **mutation-type embedding**
(initialized from BLOSUM62) that captures the physicochemical character of each
amino acid substitution.

## Architecture

```
┌─────────────────────────────────────────┐
│  Backbone                               │
│  ProteinMPNN encoder + ESM2 + adapter   │
│  Cross-attention fusion → (B, L, 1408)  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  DELTA Decomposition Head               │
│                                         │
│  StructuralContextEncoder               │
│  └─ k-nearest Cα neighbor attention     │
│  └─ Distance-weighted (learnable σ)     │
│                                         │
│  MutationTypeEmbedding                  │
│  └─ 20×20×d_sub BLOSUM62-initialized   │
│                                         │
│  ┌─────────────┐  ┌──────────────┐      │
│  │ Gain Branch  │  │ Loss Branch  │      │
│  │ mut_aa query │  │ wt_aa query  │      │
│  │ → scalar     │  │ → scalar     │      │
│  └──────┬───────┘  └──────┬───────┘      │
│         └────┐  ┌─────────┘              │
│              ▼  ▼                        │
│        ΔΔG = gain − loss                 │
└──────────────────────────────────────────┘
```

## Repository Structure

```
protstab_models/
├── base.py               # BaseModel class
├── mlp.py                # MLP head (original SPURS)
├── loss.py               # L1/MSE losses
├── spurs.py              # Original SPURS model
├── delta.py              # DELTA model (NEW)
├── decomposition.py      # Gain/Loss branches + MutTypeEmbed (NEW)
├── protein_mpnn/         # ProteinMPNN encoder
└── esm_adapter/          # ESM2 + structural adapter

protstab_train/
├── train.py              # Original SPURS training
├── evaluate.py           # Original evaluation
├── delta_train.py        # DELTA training pipeline (NEW)
├── delta_evaluate.py     # DELTA evaluation + decomposition analysis (NEW)
├── delta_loss.py         # Decomposition-aware losses (NEW)
├── custom_losses.py      # BCAS, OOD-margin losses
├── imbalanced_losses.py  # BMC, LDS losses
├── metrics.py            # Spearman, Pearson, RMSE, MAE
└── ...

protstab_data/            # Data loading (unchanged)

run_delta_ablation.py     # Paper ablation sweep (NEW)
```

## Quick Start

### 1. Training

```bash
# Basic DELTA training
python -m protstab_train.delta_train \
    --data_root /path/to/megascale_data \
    --output_dir runs/delta_v1 \
    --use_bcas \
    --use_decomp_antisym \
    --use_decomp_reg

# With BMC loss
python -m protstab_train.delta_train \
    --data_root /path/to/megascale_data \
    --output_dir runs/delta_bmc \
    --loss_type bmc \
    --use_bcas \
    --use_decomp_antisym
```

### 2. Evaluation

```bash
python -m protstab_train.delta_evaluate \
    --data_root /path/to/megascale_data \
    --ckpt runs/delta_v1/best.pt \
    --out runs/delta_v1/eval
```

This produces:
- `test_metrics.csv` — Standard metrics on all 11 held-out benchmarks
- `gain_heatmap_*.npy` / `loss_heatmap_*.npy` — 20×20 mutation-type heatmaps
- `mutation_type_metrics_*.csv` — Per-substitution-type Spearman correlations
- `learned_mut_type_embedding.npy` — Learned 20×20×d_sub embedding
- `learned_distance_scales.npy` — Learned per-head distance kernel σ values

### 3. Ablation Sweep (for paper)

```bash
python run_delta_ablation.py \
    --data_root /path/to/megascale_data \
    --output_base runs/delta_ablation \
    --max_epochs 200

# Or run specific configs:
python run_delta_ablation.py \
    --data_root /path/to/data \
    --output_base runs/ablation \
    --configs F_full G_full_bmc \
    --seeds 42 123 456
```

Ablation configurations:
| Config | Decomposition | BLOSUM Init | BCAS | Decomp Anti-Sym | Decomp Reg |
|--------|:---:|:---:|:---:|:---:|:---:|
| B | ✓ | | | | |
| C | ✓ | ✓ | | | |
| D | ✓ | ✓ | ✓ | | |
| E | ✓ | ✓ | ✓ | ✓ | |
| **F (headline)** | **✓** | **✓** | **✓** | **✓** | **✓** |
| G | ✓ | ✓ | ✓ | ✓ | ✓ + BMC |

## Loss Components

1. **Primary loss** (Huber or BMC): Standard ΔΔG regression
2. **BCAS anti-symmetry**: Bias-corrected penalty on ΔΔG(A→B) + ΔΔG(B→A)
3. **Decomposition anti-symmetry**: Direct penalty on gain/loss branch consistency
4. **Decomposition regularizer**: Prevents |gain| + |loss| >> |ΔΔG|
5. **Mutation-type smoothness** (optional): L2 on embedding gradient

## Paper Figures from Evaluation Output

The evaluation script produces all data needed for paper figures:

- **Figure 2**: `gain_heatmap_*.npy` + `loss_heatmap_*.npy` → 20×20 substitution matrices
- **Figure 3**: `mutation_type_metrics_*.csv` → per-type performance breakdown
- **Figure 4**: `learned_distance_scales.npy` → learned attention radius per head
- **Figure 5**: `learned_mut_type_embedding.npy` → t-SNE/UMAP of substitution space

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_sub` | 64 | Mutation-type embedding dimension |
| `d_aa` | 64 | Amino acid embedding dimension |
| `d_hidden` | 512 | Branch MLP hidden dimension |
| `n_branch_layers` | 3 | MLP layers per branch |
| `k_neighbors` | 32 | Spatial neighbors for context |
| `sigma_init` | 8.0 Å | Initial distance kernel scale |
| `n_context_heads` | 8 | Attention heads in context encoder |
| `decomp_reg_weight` | 0.1 | Decomposition regularizer weight |
| `decomp_reg_alpha` | 2.0 | Magnitude slack factor |
| `bcas_weight` | 0.5 | BCAS anti-symmetry weight |
| `decomp_antisym_weight` | 0.3 | Branch anti-symmetry weight |

## Requirements

Same as the existing repository:
- PyTorch ≥ 1.12
- ESM (fair-esm)
- BioPython
- scipy, numpy, pandas, tqdm

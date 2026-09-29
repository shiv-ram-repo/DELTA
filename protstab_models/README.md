# Standalone SPURS Reproduction

A complete, dependency-clean reimplementation of **SPURS** (Li & Luo, Nature
Communications 2025, [paper](https://www.nature.com/articles/s41467-025-67609-4),
[official repo](https://github.com/luo-group/SPURS)) — with **no imports from
the `spurs` package anywhere**.

Three sibling packages:

```
protstab_data/      # data loading + featurisation (see protstab_data/README.md)
protstab_models/    # neural network models (this work)
protstab_train/     # training + evaluation loops
```

## What's reproduced

| Component | Source SPURS file | Our file |
|---|---|---|
| PDB parser | `spurs/datamodules/datasets/utils.py` | `protstab_data/pdb_parser.py` |
| ProteinMPNN featurisation | `tied_featurize`, `get_pdb` (same file) | `protstab_data/featurizer.py` |
| ESM Alphabet + CoordBatchConverter | `data_utils.py`, `batch.py` | `protstab_data/featurizer.py` |
| All 11 dataset classes | `spurs/datamodules/datasets/*.py` | `protstab_data/datasets/*.py` |
| ProteinMPNN model | `org_transfer_model.py` (lines 700-1510) | `protstab_models/protein_mpnn/` |
| ESM2 + structural adapter | `modules/esm2_adapter.py` | `protstab_models/esm_adapter/` |
| MLP head | `mlp.py` | `protstab_models/mlp.py` |
| SPURS forward | `spurs.py` | `protstab_models/spurs.py` |
| L1/MSE losses | `modules/l1loss.py`, `mseloss.py` | `protstab_models/loss.py` |
| Training loop | `training_pipeline.py` + Lightning | `protstab_train/train.py` |
| Evaluation | `testing_pipeline.py` | `protstab_train/evaluate.py` |

## Installation

```bash
# 1. ESM (required for embedding model + Alphabet)
pip install git+https://github.com/facebookresearch/esm.git

# 2. Core deps
pip install torch>=1.12 numpy pandas tqdm biopython omegaconf scipy

# 3. Install the three packages in editable mode
pip install -e protstab_data
pip install -e protstab_models
# protstab_train just needs to be on PYTHONPATH

# 4. (Optional) ProteinMPNN pretrained weights
#    Download v_48_020.pt from the official ProteinMPNN repo:
#    https://github.com/dauparas/ProteinMPNN/blob/main/vanilla_model_weights/v_48_020.pt
```

## Data layout

Point `data_root` at a directory with this structure (same as SPURS expects):

```
data_root/
└── data/
    └── dataset/
        ├── megascale/
        │   ├── Tsuboyama2023_Dataset2_Dataset3_20230416.csv
        │   ├── mmseq_mut_search_0.25.m8
        │   ├── mega_splits.pkl
        │   └── AlphaFold_model_PDBs/<wt_name>.pdb
        ├── fireprot/fireprot_upload/...
        ├── ssym/...
        ├── S669/...
        └── geostab_data/
            ├── ddG_cleaned/{S461, S783, S8754, S2648}/
            └── dTm_cleaned/{S571, S4346}/
```

Download links and exact format are described in the SPURS README and paper
supplementary.

## Quick start: training

```python
from protstab_train import train, TrainConfig

cfg = TrainConfig(
    data_root        = "/path/to/spurs/data/root",
    output_dir       = "runs/spurs_repro_v1",
    proteinmpnn_ckpt = "vanilla_model_weights/v_48_020.pt",  # optional
    esm_name         = "esm2_t33_650M_UR50D",
    adapter_layer    = -3,        # = layer 31 of 33 (paper's Adapter location)
    tune_mpnn        = True,
    max_epochs       = 200,
    early_stop_pat   = 30,
    lr               = 1e-4,
)

ckpt_path = train(cfg)
```

This will:
1. Load MegaScale train/val splits (with mmseqs2 filtering)
2. Build the SPURS architecture (ProteinMPNN + ESM2 + adapter + MLP)
3. Train with L1 loss until validation Spearman stops improving
4. Save the best checkpoint to `runs/spurs_repro_v1/best.pt`

## Quick start: evaluation on all 11 benchmarks

```python
from protstab_train import evaluate_all_benchmarks

df = evaluate_all_benchmarks(
    data_root  = "/path/to/spurs/data/root",
    ckpt_path  = "runs/spurs_repro_v1/best.pt",
    output_dir = "runs/spurs_repro_v1/eval",
    adapter_layer = -3,
)
print(df)
```

Outputs:
- `runs/spurs_repro_v1/eval/test_metrics.csv` — summary table
- `runs/spurs_repro_v1/eval/preds_<name>.npy`  — per-benchmark predictions
- `runs/spurs_repro_v1/eval/targets_<name>.npy` — ground-truth labels

The 11 benchmarks reported:
`megascale_test`, `fireport_hf`, `ssym_direct`, `ssym_inverse`, `s669`,
`s461`, `s783`, `s8754`, `s2648`, `s571`, `s4346`.

## How the model behaves

`SPURS.forward(batch)` returns a `(K,)` tensor of ΔΔG predictions, one per
mutation in the batch. Pass `return_logist=True` to get the full
`(L, 20)` ΔΔG table for every position × amino acid in a single pass.

Internally:
1. ProteinMPNN encodes wild-type structure (3 enc + 3 dec layers).
2. ESM2 processes wild-type sequence; one of its transformer layers
   (default: the third-to-last, matching paper) is replaced with an
   adapter-augmented layer that cross-attends to the ProteinMPNN features.
3. ESM2's last representation is concatenated with the ProteinMPNN
   features for the L+2-token padded sequence.
4. The per-mutation representation is gathered and passed through a
   3-hidden-layer MLP outputting a 21-dim "free energy" vector per position.
5. ΔΔG = (energy[mut_aa] − energy[wt_aa]) via one-hot masking.

## Notes on faithfulness

- **One-shot decoding**: SPURS sets `use_input_decoding_order=False` and
  overrides the decoder's mask to all-ones (every residue sees every other).
  Reproduced exactly in `ProteinMPNN.forward`.
- **MLP output_dim=21**: SPURS's `MLPConfig` default is 1 in source but is
  overridden to 21 at runtime via Hydra. We set 21 as our default.
- **Frozen ESM**: All non-adapter parameters of ESM2 are frozen on load.
  Trainable parameter count should match SPURS's reported ~9.9M.
- **ProteinMPNN trainability**: SPURS sets `tune=True` for ProteinMPNN, so it
  is fine-tuned alongside the adapter. Reproduced via `cfg.encoder.tune=True`.

## File map

```
protstab_models/
├── __init__.py
├── base.py
├── mlp.py
├── loss.py
├── spurs.py                       ← top-level SPURS model
├── protein_mpnn/
│   ├── __init__.py
│   ├── config.py
│   ├── modules.py                 ← gather, EncLayer, DecLayer, ProteinFeatures
│   ├── model.py                   ← ProteinMPNN class
│   └── loader.py                  ← get_protein_mpnn()
├── esm_adapter/
│   ├── __init__.py
│   ├── adapter_layer.py           ← Transformer layer + cross-attn adapter
│   └── esm2_adapter.py            ← ESM2WithStructuralAdapter wrapper
└── setup.py

protstab_train/
├── __init__.py
├── metrics.py
├── train.py                       ← main training loop
└── evaluate.py                    ← evaluation on all 11 benchmarks
```

# RL Training for OFALMA

## Quick Start

**Train the model** (encodings are auto-computed if needed):
```bash
python training/train_ofalma_rl.py --dataset data/dataset_100.json
```

Or with more options:
```bash
python training/train_ofalma_rl.py \
    --dataset data/dataset_100.json \
    --output models/ofalma_ppo \
    --timesteps 10000 \
    --buffer-size 160
```

**Optional**: Specify a custom encodings file:
```bash
python training/train_ofalma_rl.py \
    --dataset data/dataset_100.json \
    --encodings encodings/custom.encodings.json
```

## How It Works

The environment (`ofalma_env.py`) automatically handles encodings:
1. **Auto-detects encoding file path** based on dataset hash: `encodings/dataset_{hash}.encodings.json`
2. **Checks if file exists** and matches the dataset (via hash)
3. **If exists + matches**: Loads pre-computed encodings ✓
4. **If missing or doesn't match**: Computes encodings on-the-fly and saves them

**No separate pre-computation step needed!**

## Files

- `training/ofalma_env.py`: Gymnasium environment (auto-computes/loads encodings)
- `training/train_ofalma_rl.py`: Training script

## Arguments

- `--dataset`: Path to dataset JSON file (default: `data/dataset_100.json`)
- `--output`: Output directory for model (default: `models/ofalma_ppo`)
- `--timesteps`: Total training timesteps (default: 10000)
- `--buffer-size`: Token buffer size (default: 160)
- `--train-ratio`: Train/validation split ratio (default: 0.8)
- `--model`: LLM model name (default: `gpt-4o`)
- `--encodings`: Path to pre-computed encodings file (optional)

## Encoding Format

Encodings are saved as JSON with:
- `metadata`: Dataset hash, source filename, total dialogues
- `encodings`: Dict mapping dialogue_id → {facts_count, impact_factors}

Each impact factor is [S, Q, E] for each fact (R is computed from position).

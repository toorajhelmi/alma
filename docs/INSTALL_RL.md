# Installing RL Dependencies

## Issue
PyTorch (required for stable-baselines3) does not support Python 3.13 yet.
You need Python 3.11 or 3.12.

## Quick Fix

### Check if Python 3.12 is available:
```bash
python3.12 --version
```

### If Python 3.12 is available:

1. **Remove old venv** (optional):
   ```bash
   rm -rf venv
   ```

2. **Create new venv with Python 3.12**:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. **Install all dependencies**:
   ```bash
   pip install -r requirements/rl.txt
   ```

### If Python 3.12 is NOT available:

**Install Python 3.12 using Homebrew:**
```bash
brew install python@3.12
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements/rl.txt
```

**Or use pyenv:**
```bash
brew install pyenv
pyenv install 3.12.7
pyenv local 3.12.7
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/rl.txt
```

## Verify Installation

After installing, verify PyTorch works:
```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

## Then Start Training

```bash
python training/train_falma_rl.py --dataset data/dataset_100.json
```

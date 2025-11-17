#!/usr/bin/env python3
"""Extract current weights from training model."""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import glob
from stable_baselines3 import PPO
import numpy as np

def find_latest_run():
    """Find the latest training run directory."""
    runs = sorted(glob.glob("models/ofalma_rate_distortion/run_*/"))
    if not runs:
        print("No training runs found")
        return None
    return Path(runs[-1])

def extract_weights_from_checkpoint(checkpoint_path):
    """Extract weights from a checkpoint file."""
    try:
        model = PPO.load(str(checkpoint_path))
        obs_shape = model.observation_space.shape
        dummy_obs = np.zeros((1,) + obs_shape, dtype=np.float32)
        action, _ = model.predict(dummy_obs, deterministic=True)
        return action[0], model.num_timesteps
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None, None

def main():
    latest_run = find_latest_run()
    if not latest_run:
        return
    
    print(f"Latest run: {latest_run}")
    print()
    
    # Try checkpoints first
    checkpoints = sorted(glob.glob(str(latest_run / "checkpoints" / "*.zip")))
    if checkpoints:
        checkpoint_path = Path(checkpoints[-1])
        weights, timesteps = extract_weights_from_checkpoint(checkpoint_path)
        if weights is not None:
            print("=" * 70)
            print(f"CURRENT WEIGHTS (from checkpoint at step {timesteps}):")
            print("=" * 70)
            print(f"  theta_S (Surprisal):  {weights[0]:.6f}")
            print(f"  theta_R (Recency):    {weights[1]:.6f}")
            print(f"  theta_Q (Relevance):  {weights[2]:.6f}")
            print(f"  theta_E (Emphasis):   {weights[3]:.6f}")
            print("=" * 70)
            print()
            
            # Save to file
            output_file = latest_run / "current_weights.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "step": int(timesteps),
                    "theta_S": float(weights[0]),
                    "theta_R": float(weights[1]),
                    "theta_Q": float(weights[2]),
                    "theta_E": float(weights[3]),
                }, f, indent=2)
            print(f"Saved to: {output_file}")
            return
    
    # Try latest_model.zip
    latest_model = latest_run / "latest_model.zip"
    if latest_model.exists():
        weights, timesteps = extract_weights_from_checkpoint(latest_model)
        if weights is not None:
            print("=" * 70)
            print(f"CURRENT WEIGHTS (from latest_model.zip at step {timesteps}):")
            print("=" * 70)
            print(f"  theta_S (Surprisal):  {weights[0]:.6f}")
            print(f"  theta_R (Recency):    {weights[1]:.6f}")
            print(f"  theta_Q (Relevance):  {weights[2]:.6f}")
            print(f"  theta_E (Emphasis):   {weights[3]:.6f}")
            print("=" * 70)
            return
    
    # Fallback: show last logged weights
    metrics_file = latest_run / "logs" / "training_metrics.jsonl"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
        if lines:
            last_entry = json.loads(lines[-1])
            if 'theta_mean' in last_entry and last_entry['theta_mean']:
                theta = last_entry['theta_mean']
                step = last_entry.get('step', 'N/A')
                print("=" * 70)
                print(f"LAST LOGGED WEIGHTS (from step {step} - may be outdated):")
                print("=" * 70)
                print(f"  theta_S: {theta[0]:.6f}")
                print(f"  theta_R: {theta[1]:.6f}")
                print(f"  theta_Q: {theta[2]:.6f}")
                print(f"  theta_E: {theta[3]:.6f}")
                print("=" * 70)
                print()
                print("⚠️  WARNING: These weights are from an earlier step!")
                print("   Checkpoints are saved every 2000 steps.")
                print("   Next checkpoint will be at step 6000.")
                return
    
    print("No weights found. Training may still be initializing.")

if __name__ == "__main__":
    main()


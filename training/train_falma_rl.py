"""Train FALMA coefficients using PPO."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain_openai import ChatOpenAI
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import torch.nn as nn
import sys
from tqdm.auto import tqdm
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.falma_env import FALMAEnv
from main import OPENAI_API_KEY
from llm import get_provider, TogetherAI, Gpt

TOGETHER_API_KEY = "4164d6f1ac5f4fe46afde0cd1506316a50d700107897538642c83cd016b470a7"


def load_dataset(filename: str) -> List[Dict]:
    """Load dialogues from JSON file."""
    import os
    if not os.path.isabs(filename) and not filename.startswith('data/'):
        filename = os.path.join('data', filename)
    with open(filename, 'r') as f:
        dialogues = json.load(f)
    print(f"Loaded {len(dialogues)} dialogues from {filename}")
    return dialogues


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Locate the most recent checkpoint file in the output directory."""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    candidates = []
    if checkpoint_dir.exists():
        candidates = sorted(
            checkpoint_dir.glob("ppo_falma_*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if candidates:
        return str(candidates[0])
    latest_path = Path(output_dir) / "latest_model.zip"
    if latest_path.exists():
        return str(latest_path)
    return None


def write_run_metadata(output_dir: str, metadata: Dict[str, Any]) -> None:
    """Persist run configuration for reproducibility."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "timestamp": timestamp,
        **metadata,
    }
    metadata_path = log_dir / "training_metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except Exception:
            existing = []
    else:
        existing = []
    if not isinstance(existing, list):
        existing = [existing]
    existing.append(payload)
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)


def _get_current_learning_rate(model: PPO) -> Optional[float]:
    """Fetch current learning rate from PPO optimizer if available."""
    try:
        optimizer = getattr(model.policy, "optimizer", None)
        if optimizer and optimizer.param_groups:
            return float(optimizer.param_groups[0].get("lr", None))
    except Exception:
        pass
    return None


def _set_learning_rate(model: PPO, new_lr: float) -> None:
    """Force PPO instance to use a constant learning rate."""
    model.learning_rate = new_lr
    model.lr_schedule = lambda _: new_lr
    optimizer = getattr(model.policy, "optimizer", None)
    if optimizer:
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


def save_training_state(output_dir: str, model: PPO, target_timesteps: int, extra_state: Optional[Dict[str, Any]] = None) -> None:
    """Persist lightweight metadata about training progress."""
    state_path = Path(output_dir) / "training_state.json"
    state = {
        "target_timesteps": target_timesteps,
        "timesteps_completed": getattr(model, "num_timesteps", None),
    }
    current_lr = _get_current_learning_rate(model)
    if current_lr is not None:
        state["current_learning_rate"] = current_lr
    if extra_state:
        state.update(extra_state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2)


def split_dataset(dialogues: List[Dict], train_ratio: float = 0.8):
    """Split dataset into train and validation sets."""
    split_idx = int(len(dialogues) * train_ratio)
    train_dialogues = dialogues[:split_idx]
    val_dialogues = dialogues[split_idx:]
    print(f"Split: {len(train_dialogues)} train, {len(val_dialogues)} validation")
    return train_dialogues, val_dialogues


def create_env(dialogues: List[Dict], llm: Optional[ChatOpenAI] = None, buffer_size: int = 160, 
               encodings_file: Optional[str] = None, model_name: Optional[str] = None, 
               api_key: Optional[str] = None, use_rate_distortion: bool = False,
               condensation_model: Optional[str] = None, evaluation_model: Optional[str] = None,
               impact_model: Optional[str] = None, token_model: Optional[str] = None):
    """Create FALMA environment (encodings will be auto-loaded/computed)."""
    # Register models in this process (important for subprocess environments)
    provider = get_provider()
    if model_name and api_key:
        # Register the default model
        try:
            gpt_llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)
            model_name_short = model_name.replace('-', '').replace('_', '').lower()
            if not provider.is_available(model_name_short):
                provider.register(model_name_short, Gpt(gpt_llm), default_remote=True)
        except Exception as e:
            print(f"Warning: Could not register model in subprocess: {e}")
    
    # Register additional models if specified
    if evaluation_model and evaluation_model != model_name_short:
        try:
            if not provider.is_available(evaluation_model):
                eval_llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)
                provider.register(evaluation_model, Gpt(eval_llm), default_remote=False)
        except Exception as e:
            print(f"Warning: Could not register evaluation model in subprocess: {e}")
    
    # Register condensation model for rate-distortion in subprocess
    if use_rate_distortion and condensation_model:
        if condensation_model == 'llama31_8b':
            try:
                if not provider.is_available('llama31_8b'):
                    llama_llm = TogetherAI(api_key=TOGETHER_API_KEY, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
                    provider.register('llama31_8b', llama_llm, default_remote=False)
            except Exception as e:
                print(f"Warning: Could not register llama31_8b in subprocess: {e}")
        elif condensation_model != model_name_short:
            # Register custom condensation model if different from default
            try:
                if not provider.is_available(condensation_model):
                    cond_llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)
                    provider.register(condensation_model, Gpt(cond_llm), default_remote=False)
            except Exception as e:
                print(f"Warning: Could not register condensation model in subprocess: {e}")
    
    if llm is None:
        if model_name is None or api_key is None:
            raise ValueError("Either llm or both model_name and api_key must be provided")
        llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)
    env = FALMAEnv(dialogues, llm, buffer_size=buffer_size, encodings_file=encodings_file, 
                   use_rate_distortion=use_rate_distortion, condensation_model=condensation_model,
                   evaluation_model=evaluation_model, impact_model=impact_model, token_model=token_model)
    return env


def train_ppo(
    train_dialogues: List[Dict],
    val_dialogues: List[Dict],
    llm: ChatOpenAI,
    output_dir: str = "models/falma_ppo",
    total_timesteps: int = 10000,
    buffer_size: int = 160,
    learning_rate: float = 1e-3,
    n_steps: int = 128,
    batch_size: int = 64,
    n_epochs: int = 10,
    encodings_file: Optional[str] = None,
    n_envs: int = 4,
    use_subproc: bool = True,
    use_rate_distortion: bool = False,
    condensation_model: Optional[str] = None,
    evaluation_model: Optional[str] = None,
    impact_model: Optional[str] = None,
    token_model: Optional[str] = None,
    resume_path: Optional[str] = None,
    resume_learning_rate: Optional[float] = None,
    lr_decay_factor: float = 0.7,
    lr_patience: int = 2,
    lr_threshold: float = 0.5,
    min_learning_rate: float = 1e-5,
):
    """Train PPO agent to learn FALMA coefficients."""
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = logs_dir / "training_metrics.jsonl"
    
    # Compute encodings for full dataset once (train + validation)
    all_dialogues = train_dialogues + val_dialogues
    from training.falma_env import FALMAEnv
    temp_env = FALMAEnv(all_dialogues, llm, buffer_size=buffer_size, encodings_file=encodings_file,
                        impact_model=impact_model)
    temp_env._load_or_compute_encodings(encodings_file)
    encodings_file = temp_env._get_encodings_file_path(encodings_file)
    temp_env = None
    
    print(f"Creating {n_envs} parallel environments...")
    
    model_name = llm.model_name if hasattr(llm, 'model_name') else 'gpt-4o'
    api_key = llm.openai_api_key if hasattr(llm, 'openai_api_key') else OPENAI_API_KEY
    
    # Compute model_name_short for subprocess registration
    model_name_short = model_name.replace('-', '').replace('_', '').lower()
    
    # Determine condensation model for subprocesses
    subprocess_condensation_model = condensation_model
    if use_rate_distortion and subprocess_condensation_model is None:
        subprocess_condensation_model = model_name_short
    
    def make_env(rank: int):
        def _init():
            env = create_env(train_dialogues, llm=None, buffer_size=buffer_size, encodings_file=encodings_file,
                           model_name=model_name, api_key=api_key, use_rate_distortion=use_rate_distortion,
                           condensation_model=subprocess_condensation_model, evaluation_model=evaluation_model or model_name_short,
                           impact_model=impact_model, token_model=token_model)
            env = Monitor(env, filename=None, allow_early_resets=True)
            return env
        return _init
    
    def make_val_env(rank: int):
        def _init():
            env = create_env(val_dialogues, llm=None, buffer_size=buffer_size, encodings_file=encodings_file,
                           model_name=model_name, api_key=api_key, use_rate_distortion=use_rate_distortion,
                           condensation_model=subprocess_condensation_model, evaluation_model=evaluation_model or model_name_short,
                           impact_model=impact_model, token_model=token_model)
            # Set seed for deterministic validation
            env._episode_counter = 0
            env = Monitor(env, filename=None, allow_early_resets=True)
            return env
        return _init
    
    if use_subproc and n_envs > 1:
        train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        val_env = SubprocVecEnv([make_val_env(i) for i in range(min(n_envs, 2))])
    else:
        train_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
        val_env = DummyVecEnv([make_val_env(i) for i in range(min(n_envs, 2))])
    
    from core.falma import theta
    initial_coeffs = np.array([
        theta["S"],
        theta["R"],
        theta["Q"],
        theta["E"]
    ], dtype=np.float32)
    
    print(f"Initial coefficients: {initial_coeffs}")
    print("Creating PPO model...")
    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}")
    
    class FlattenFeatureExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space):
            n_flatten = observation_space.shape[0] * observation_space.shape[1]
            super().__init__(observation_space, features_dim=256)
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(n_flatten, 256)
            
        def forward(self, observations):
            return self.linear(self.flatten(observations))
    
    policy_kwargs = dict(
        features_extractor_class=FlattenFeatureExtractor,
        features_extractor_kwargs=dict(),
    )
    
    start_timesteps = 0
    model = None

    if resume_path and Path(resume_path).exists():
        print(f"Resuming training from checkpoint: {resume_path}")
        model = PPO.load(
            resume_path,
            env=train_env,
            device="auto",
        )
        model.tensorboard_log = f"{output_dir}/tensorboard"
        start_timesteps = getattr(model, "num_timesteps", 0)
        print(f"Checkpoint timesteps: {start_timesteps}/{total_timesteps}")
        if resume_learning_rate is not None:
            print(f"Overriding learning rate on resume to {resume_learning_rate:.2e}")
            _set_learning_rate(model, resume_learning_rate)
        else:
            current_lr = _get_current_learning_rate(model)
            if current_lr is not None:
                print(f"Resumed with optimizer learning rate {current_lr:.2e}")

    if model is None:
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            verbose=1,
            tensorboard_log=f"{output_dir}/tensorboard",
        )
        print(f"Initial learning rate: {learning_rate:.2e}")
    
    print(f"Using {n_envs} parallel environments (n_steps={n_steps}, collecting {n_steps * n_envs} steps per update)")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=f"{output_dir}/checkpoints",
        name_prefix="ppo_falma",
    )
    
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=f"{output_dir}/best",
        log_path=f"{output_dir}/logs",
        eval_freq=500,  # Reduced from 100 to save on LLM API costs (80% reduction)
        n_eval_episodes=len(val_dialogues),
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    class ProgressWithValidationCallback(BaseCallback):
        def __init__(
            self,
            print_freq: int = 50,
            eval_callback=None,
            verbose: int = 1,
            n_steps_per_update: int = 128,
            start_offset: int = 0,
            progress_bar_callback: Optional["TqdmProgressCallback"] = None,
            min_val_accuracy: float = 45.0,
            metrics_log_path: Optional[str] = None,
            lr_decay_factor: float = 1.0,
            lr_patience: int = 2,
            lr_threshold: float = 0.5,
            min_learning_rate: float = 1e-5,
            initial_learning_rate: float = 3e-4,
            eval_freq: int = 500,
        ):
            super().__init__(verbose)
            self.eval_callback = eval_callback
            self.episode_infos = []  # Store info dicts from completed episodes
            self._last_printed_episode_count = 0
            self.start_offset = start_offset
            self._last_line_length = 0
            self._best_train_acc = 0.0
            self.progress_bar_callback = progress_bar_callback
            self.min_val_accuracy = min_val_accuracy
            self._best_val_acc: Optional[float] = None
            self._last_eval_reward: Optional[float] = None
            self._last_eval_step: Optional[int] = None
            self._last_logged_step: int = 0
            self._theta_history: List[np.ndarray] = []
            self._theta_history_window = 500
            self.latest_theta_mean: Optional[np.ndarray] = None
            self.metrics_log_path = Path(metrics_log_path) if metrics_log_path else None
            self.lr_decay_factor = lr_decay_factor if lr_decay_factor > 0 else 1.0
            self.lr_patience = max(1, lr_patience)
            self.lr_threshold = max(0.0, lr_threshold)
            self.min_learning_rate = max(0.0, min_learning_rate)
            self.initial_learning_rate = initial_learning_rate
            self._no_improve_evals = 0
            self.eval_freq = eval_freq
            self._eval_history: List[Dict[str, Any]] = []  # Track eval history for plateau detection

        def _log_metrics(
            self,
            step: int,
            total_episodes: int,
            train_accuracy: float,
            recent_correct: List[int],
            avg_coeffs: Optional[Dict[str, float]],
            val_acc: Optional[float],
            best_val_acc: Optional[float],
            current_lr: Optional[float],
            force_log: bool = False,
        ) -> None:
            """Log metrics. If force_log=True, always log even if step hasn't changed."""
            if not self.metrics_log_path:
                return
            
            # Only log if this is an eval step (every eval_freq steps) or forced
            is_eval_step = (step % self.eval_freq == 0) and step > 0
            if not force_log and not is_eval_step and step == self._last_logged_step:
                return
            
            entry: Dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": int(step),
                "episodes": int(total_episodes),
                "recent_window": len(recent_correct),
                "train_accuracy": float(train_accuracy),
                "best_train_accuracy": float(self._best_train_acc),
                "is_eval_step": is_eval_step,
            }
            
            # Always include weights (theta_mean) - critical for paper
            if avg_coeffs:
                entry["theta_average"] = {
                    "S": float(avg_coeffs.get("theta_S", 0.0)),
                    "R": float(avg_coeffs.get("theta_R", 0.0)),
                    "Q": float(avg_coeffs.get("theta_Q", 0.0)),
                    "E": float(avg_coeffs.get("theta_E", 0.0)),
                }
            if self.latest_theta_mean is not None:
                entry["theta_mean"] = [float(v) for v in self.latest_theta_mean.tolist()]
            else:
                # If no theta_mean yet, use average from recent episodes
                if avg_coeffs:
                    entry["theta_mean"] = [
                        float(avg_coeffs.get("theta_S", 0.0)),
                        float(avg_coeffs.get("theta_R", 0.0)),
                        float(avg_coeffs.get("theta_Q", 0.0)),
                        float(avg_coeffs.get("theta_E", 0.0)),
                    ]
            
            # Always include validation accuracy if available (critical for paper)
            if val_acc is not None:
                entry["validation_accuracy"] = float(val_acc)
            if best_val_acc is not None:
                entry["best_validation_accuracy"] = float(best_val_acc)
            
            # Learning rate information
            if current_lr is not None:
                entry["learning_rate"] = float(current_lr)
            entry["initial_learning_rate"] = float(self.initial_learning_rate)
            entry["lr_decay_factor"] = float(self.lr_decay_factor)
            entry["lr_patience"] = int(self.lr_patience)
            entry["lr_threshold"] = float(self.lr_threshold)
            entry["min_learning_rate"] = float(self.min_learning_rate)
            entry["no_improve_evals"] = int(self._no_improve_evals)
            
            try:
                with self.metrics_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
                self._last_logged_step = step
            except Exception as exc:
                print(f"\n[Warning] Could not log metrics: {exc}")
            
        def _on_step(self) -> bool:
            if self.locals.get('dones') is not None:
                dones = self.locals['dones']
                infos = self.locals.get('infos', [])
                rewards = self.locals.get('rewards', [])
                
                for i, done in enumerate(dones):
                    if done:
                        info = infos[i] if i < len(infos) else {}
                        reward = rewards[i] if i < len(rewards) else 0.0
                        
                        if isinstance(info, dict):
                            self.episode_infos.append(info.copy())
                            # Keep only last 1000 episodes to prevent memory growth
                            if len(self.episode_infos) > 1000:
                                self.episode_infos = self.episode_infos[-1000:]
                            if all(k in info for k in ("theta_S", "theta_R", "theta_Q", "theta_E")):
                                theta_vec = np.array(
                                    [
                                        float(info.get("theta_S", 0.0)),
                                        float(info.get("theta_R", 0.0)),
                                        float(info.get("theta_Q", 0.0)),
                                        float(info.get("theta_E", 0.0)),
                                    ],
                                    dtype=np.float32,
                                )
                                self._theta_history.append(theta_vec)
                                if len(self._theta_history) > self._theta_history_window:
                                    self._theta_history = self._theta_history[-self._theta_history_window:]
            
            # Print whenever we have new episodes
            has_new_episodes = len(self.episode_infos) > self._last_printed_episode_count
            
            # Check if we're at an eval step (every eval_freq steps) - ALWAYS check this
            current_step = self.model.num_timesteps if self.model is not None else self.num_timesteps
            is_eval_step = (current_step % self.eval_freq == 0) and current_step > 0
            
            avg_coeffs: Optional[Dict[str, float]] = None
            val_acc_value: Optional[float] = None

            # Always process if we have episodes OR if it's an eval step
            if len(self.episode_infos) > 0 and (has_new_episodes or is_eval_step):
                recent_window = min(100, len(self.episode_infos))
                recent_infos = self.episode_infos[-recent_window:]
                
                recent_correct = [1 if info.get('correct', False) else 0 for info in recent_infos]
                train_accuracy = sum(recent_correct) / len(recent_correct) * 100 if recent_correct else 0.0
                if train_accuracy > self._best_train_acc:
                    self._best_train_acc = train_accuracy
                total_episodes = len(self.episode_infos)
                
                if self._theta_history:
                    theta_window = min(len(self._theta_history), recent_window)
                    theta_array = np.vstack(self._theta_history[-theta_window:])
                    self.latest_theta_mean = theta_array.mean(axis=0)
                else:
                    self.latest_theta_mean = None
                
                # Calculate average coefficients and threshold from info dicts
                coeffs_text = ""
                if recent_infos:
                    avg_coeffs = {
                        'theta_S': sum(float(info.get('theta_S', 0.0)) for info in recent_infos) / len(recent_infos),
                        'theta_R': sum(float(info.get('theta_R', 0.0)) for info in recent_infos) / len(recent_infos),
                        'theta_Q': sum(float(info.get('theta_Q', 0.0)) for info in recent_infos) / len(recent_infos),
                        'theta_E': sum(float(info.get('theta_E', 0.0)) for info in recent_infos) / len(recent_infos),
                    }
                    coeffs_text = (f" | θ_S={avg_coeffs['theta_S']:.2f} θ_R={avg_coeffs['theta_R']:.2f} "
                                 f"θ_Q={avg_coeffs['theta_Q']:.2f} θ_E={avg_coeffs['theta_E']:.2f}")
                
                val_text = ""
                current_lr = _get_current_learning_rate(self.model) if self.model is not None else None
                
                if self.eval_callback and hasattr(self.eval_callback, 'last_mean_reward'):
                    try:
                        reward = float(self.eval_callback.last_mean_reward)
                        # Reward is [-1, 1], convert to accuracy [0, 100%]
                        if not (np.isinf(reward) or np.isnan(reward)) and -1 <= reward <= 1:
                            val_acc = (reward + 1) / 2 * 100
                            val_text = f" | Val Accuracy: {val_acc:.1f}%"
                            val_acc_value = val_acc
                            
                            # Track eval history for plateau detection
                            if is_eval_step or self._last_eval_step != current_step:
                                self._eval_history.append({
                                    "step": current_step,
                                    "val_acc": val_acc,
                                    "reward": reward,
                                })
                                # Keep only last 20 evals for plateau detection
                                if len(self._eval_history) > 20:
                                    self._eval_history = self._eval_history[-20:]
                            
                            if self._last_eval_reward != reward or current_step != self._last_eval_step:
                                print(f"\n[Eval] step {current_step} | validation accuracy {val_acc:.1f}% (= reward {reward:.3f})")
                                self._last_eval_reward = reward
                                self._last_eval_step = current_step
                                
                                # CRITICAL: Log immediately when eval completes with weights and eval accuracy
                                # Get current weights
                                if len(self.episode_infos) > 0:
                                    recent_window = min(100, len(self.episode_infos))
                                    recent_infos = self.episode_infos[-recent_window:]
                                    if recent_infos:
                                        avg_coeffs_eval = {
                                            'theta_S': sum(float(info.get('theta_S', 0.0)) for info in recent_infos) / len(recent_infos),
                                            'theta_R': sum(float(info.get('theta_R', 0.0)) for info in recent_infos) / len(recent_infos),
                                            'theta_Q': sum(float(info.get('theta_Q', 0.0)) for info in recent_infos) / len(recent_infos),
                                            'theta_E': sum(float(info.get('theta_E', 0.0)) for info in recent_infos) / len(recent_infos),
                                        }
                                        recent_correct_eval = [1 if info.get('correct', False) else 0 for info in recent_infos]
                                        train_acc_eval = sum(recent_correct_eval) / len(recent_correct_eval) * 100 if recent_correct_eval else 0.0
                                    else:
                                        avg_coeffs_eval = None
                                        train_acc_eval = 0.0
                                        recent_correct_eval = []
                                else:
                                    avg_coeffs_eval = None
                                    train_acc_eval = 0.0
                                    recent_correct_eval = []
                                
                                # Update theta_mean
                                if self._theta_history:
                                    theta_window = min(len(self._theta_history), 100)
                                    theta_array = np.vstack(self._theta_history[-theta_window:])
                                    self.latest_theta_mean = theta_array.mean(axis=0)
                                
                                # Log with eval accuracy
                                self._log_metrics(
                                    step=current_step,
                                    total_episodes=len(self.episode_infos),
                                    train_accuracy=train_acc_eval,
                                    recent_correct=recent_correct_eval,
                                    avg_coeffs=avg_coeffs_eval,
                                    val_acc=val_acc,
                                    best_val_acc=self._best_val_acc,
                                    current_lr=current_lr,
                                    force_log=True,
                                )

                                if self._best_val_acc is None:
                                    if current_step >= self.eval_freq and val_acc < self.min_val_accuracy:
                                        print(f"[Early stop] validation accuracy {val_acc:.1f}% below minimum {self.min_val_accuracy:.1f}% at step {current_step}.")
                                        self._log_metrics(
                                            step=current_step,
                                            total_episodes=total_episodes,
                                            train_accuracy=train_accuracy,
                                            recent_correct=recent_correct,
                                            avg_coeffs=avg_coeffs,
                                            val_acc=val_acc,
                                            best_val_acc=self._best_val_acc,
                                            current_lr=current_lr,
                                            force_log=True,
                                        )
                                        return False
                                    self._best_val_acc = val_acc
                                    self._no_improve_evals = 0
                                else:
                                    improvement = val_acc - self._best_val_acc
                                    if improvement > self.lr_threshold:
                                        self._best_val_acc = val_acc
                                        self._no_improve_evals = 0
                                    else:
                                        self._no_improve_evals += 1
                                        
                                    # Adaptive learning rate: reduce when plateauing
                                    reduced = False
                                    if current_lr is not None and current_lr > self.min_learning_rate + 1e-12 and self.lr_decay_factor < 1.0:
                                        # Check if we've plateaued (no significant improvement for lr_patience evals)
                                        if self._no_improve_evals >= self.lr_patience:
                                            new_lr = max(self.min_learning_rate, current_lr * self.lr_decay_factor)
                                            if new_lr < current_lr - 1e-12:
                                                print(f"[LR] Plateau detected at step {current_step} (no improvement for {self._no_improve_evals} evals); reducing learning rate {current_lr:.2e} -> {new_lr:.2e}")
                                                _set_learning_rate(self.model, new_lr)
                                                current_lr = new_lr
                                                reduced = True
                                                self._no_improve_evals = 0  # Reset counter after LR reduction
                                        
                                    if not reduced and current_lr is not None and current_lr <= self.min_learning_rate + 1e-12 and val_acc <= self._best_val_acc:
                                        print(f"[Early stop] validation accuracy {val_acc:.1f}% did not improve over best {self._best_val_acc:.1f}% with learning rate at minimum.")
                                        self._log_metrics(
                                            step=current_step,
                                            total_episodes=total_episodes,
                                            train_accuracy=train_accuracy,
                                            recent_correct=recent_correct,
                                            avg_coeffs=avg_coeffs,
                                            val_acc=val_acc,
                                            best_val_acc=self._best_val_acc,
                                            current_lr=current_lr,
                                            force_log=True,
                                        )
                                        return False
                    except (ValueError, TypeError):
                        pass
                
            # CRITICAL: Always log at eval steps, even if no new episodes
            # This ensures we log every 500 steps regardless of episode completion timing
            if is_eval_step:
                # Get current metrics even if no new episodes
                if len(self.episode_infos) == 0:
                    # No episodes yet, use defaults
                    total_episodes = 0
                    train_accuracy = 0.0
                    recent_correct = []
                    avg_coeffs = None
                else:
                    # Use existing calculated values or recalculate
                    if avg_coeffs is None:
                        recent_window = min(100, len(self.episode_infos))
                        recent_infos = self.episode_infos[-recent_window:]
                        if recent_infos:
                            avg_coeffs = {
                                'theta_S': sum(float(info.get('theta_S', 0.0)) for info in recent_infos) / len(recent_infos),
                                'theta_R': sum(float(info.get('theta_R', 0.0)) for info in recent_infos) / len(recent_infos),
                                'theta_Q': sum(float(info.get('theta_Q', 0.0)) for info in recent_infos) / len(recent_infos),
                                'theta_E': sum(float(info.get('theta_E', 0.0)) for info in recent_infos) / len(recent_infos),
                            }
                        recent_correct = [1 if info.get('correct', False) else 0 for info in recent_infos]
                        train_accuracy = sum(recent_correct) / len(recent_correct) * 100 if recent_correct else 0.0
                    total_episodes = len(self.episode_infos)
                
                # Get validation accuracy if available - try multiple sources
                if val_acc_value is None:
                    # First try eval callback
                    if self.eval_callback and hasattr(self.eval_callback, 'last_mean_reward'):
                        try:
                            reward = float(self.eval_callback.last_mean_reward)
                            if not (np.isinf(reward) or np.isnan(reward)) and -1 <= reward <= 1:
                                val_acc_value = (reward + 1) / 2 * 100
                        except:
                            pass
                    
                    # If still None, try reading from evaluations.npz file directly
                    if val_acc_value is None and self.metrics_log_path:
                        try:
                            eval_file = self.metrics_log_path.parent / "evaluations.npz"
                            if eval_file.exists():
                                eval_data = np.load(eval_file)
                                eval_timesteps = eval_data['timesteps']
                                eval_results = eval_data['results']
                                eval_accuracies = ((eval_results + 1) / 2).mean(axis=1) * 100
                                
                                # Find the most recent evaluation at or before current_step
                                for i in range(len(eval_timesteps) - 1, -1, -1):
                                    if eval_timesteps[i] <= current_step:
                                        val_acc_value = float(eval_accuracies[i])
                                        break
                        except Exception:
                            pass
                
                # Update theta_mean if we have history
                if self._theta_history:
                    theta_window = min(len(self._theta_history), 100)
                    theta_array = np.vstack(self._theta_history[-theta_window:])
                    self.latest_theta_mean = theta_array.mean(axis=0)
                
                # Force log at eval step
                self._log_metrics(
                    step=current_step,
                    total_episodes=total_episodes,
                    train_accuracy=train_accuracy,
                    recent_correct=recent_correct,
                    avg_coeffs=avg_coeffs,
                    val_acc=val_acc_value,
                    best_val_acc=self._best_val_acc,
                    current_lr=current_lr,
                    force_log=True,
                )
                
                best_val_text = ""
                if self._best_val_acc is not None:
                    best_val_text = f" | Val Best {self._best_val_acc:.1f}%"
                theta_text = ""
                if self.latest_theta_mean is not None:
                    theta_text = " | theta_mean=" + ",".join(f"{v:.2f}" for v in self.latest_theta_mean.tolist())
                status_text = (
                    f"step {current_step} | episodes {total_episodes} | "
                    f"train {train_accuracy:.1f}% [{self._best_train_acc:.1f}%] "
                    f"({sum(recent_correct)}/{len(recent_correct)})"
                    f"{coeffs_text}{val_text}{best_val_text}{theta_text}"
                )
                if current_lr is not None:
                    status_text += f" | lr={current_lr:.2e}"

                if self.progress_bar_callback and self.progress_bar_callback.set_status(status_text):
                    pass
                else:
                    line = f"\r[{status_text}]"
                    sys.stdout.write(line)
                    visible_length = len(line) - 1 if line.startswith("\r") else len(line)
                    extra_spaces = max(0, self._last_line_length - visible_length)
                    if extra_spaces:
                        sys.stdout.write(" " * extra_spaces)
                    sys.stdout.flush()
                    self._last_line_length = visible_length
                self._last_printed_episode_count = len(self.episode_infos)
            return True

        def _on_training_end(self) -> None:
            if self.progress_bar_callback:
                self.progress_bar_callback.clear_status()
            if self._last_line_length:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._last_line_length = 0
        
        def get_state_snapshot(self) -> Dict[str, Any]:
            snapshot: Dict[str, Any] = {}
            if self._best_val_acc is not None:
                snapshot["best_val_accuracy"] = self._best_val_acc
            if self.latest_theta_mean is not None:
                snapshot["latest_theta_mean"] = [float(v) for v in self.latest_theta_mean]
            current_lr = _get_current_learning_rate(self.model) if self.model is not None else None
            if current_lr is not None:
                snapshot["current_learning_rate"] = current_lr
            return snapshot
    
    class TqdmProgressCallback(BaseCallback):
        def __init__(self, total: int, initial_steps: int = 0):
            super().__init__()
            self.total = total
            self.initial_steps = initial_steps
            self._pbar: Optional[tqdm] = None
            self._last_update = initial_steps
            self._status = ""

        def _on_training_start(self) -> None:
            initial = self.initial_steps
            if self.model is not None:
                initial = self.model.num_timesteps
            self._last_update = initial
            self._pbar = tqdm(total=self.total, initial=initial, desc="Timesteps", unit="step")
            if self._status:
                self._pbar.set_postfix_str(self._status, refresh=True)

        def _on_step(self) -> bool:
            if self._pbar is None:
                return True
            current = self.model.num_timesteps if self.model is not None else self._last_update
            if current > self.total:
                current = self.total
            if current > self._last_update:
                self._pbar.n = current
                self._pbar.refresh()
                self._last_update = current
            return True

        def _on_training_end(self) -> None:
            if self._pbar is not None:
                current = self.model.num_timesteps if self.model is not None else self._last_update
                self._pbar.n = min(self.total, current)
                self._pbar.refresh()
                self._pbar.close()

        def set_status(self, text: str) -> bool:
            self._status = text
            if self._pbar is not None:
                self._pbar.set_postfix_str(text, refresh=True)
            return True

        def clear_status(self) -> None:
            self._status = ""
            if self._pbar is not None:
                self._pbar.set_postfix_str("", refresh=True)

    current_timesteps = model.num_timesteps if hasattr(model, "num_timesteps") else 0
    progress_bar_callback = TqdmProgressCallback(total=total_timesteps, initial_steps=current_timesteps)

    progress_callback = ProgressWithValidationCallback(
        print_freq=10,  # Not used anymore, kept for compatibility
        eval_callback=eval_callback,
        n_steps_per_update=n_steps * n_envs,  # Not used anymore, kept for compatibility
        start_offset=start_timesteps,
        progress_bar_callback=progress_bar_callback,
        min_val_accuracy=45.0,
        metrics_log_path=str(metrics_log_path),
        lr_decay_factor=lr_decay_factor,
        lr_patience=lr_patience,
        lr_threshold=lr_threshold,
        min_learning_rate=min_learning_rate,
        initial_learning_rate=learning_rate,
        eval_freq=500,
    )

    remaining_timesteps = max(0, total_timesteps - start_timesteps)
    if remaining_timesteps == 0:
        print("Requested timesteps already achieved. Skipping additional training.")
        latest_model_path = Path(output_dir) / "latest_model"
        model.save(latest_model_path)
        save_training_state(output_dir, model, total_timesteps, extra_state=progress_callback.get_state_snapshot())
        print(f"Current model saved to {latest_model_path}")
        return model

    print(f"Starting training for {remaining_timesteps} additional timesteps (target={total_timesteps})...")
    print(f"Train dialogues: {len(train_dialogues)}")
    print(f"Validation dialogues: {len(val_dialogues)}")
    print(f"Each episode = 1 dialogue")
    print(f"Total episodes so far: ~{start_timesteps} | Remaining episodes: ~{remaining_timesteps}")
    
    callback_list = CallbackList([eval_callback, checkpoint_callback, progress_callback, progress_bar_callback])

    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callback_list,
            progress_bar=False,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving latest checkpoint...")
        progress_bar_callback._on_training_end()
        progress_callback._on_training_end()
        latest_model_path = Path(output_dir) / "latest_model"
        model.save(latest_model_path)
        save_training_state(output_dir, model, total_timesteps, extra_state=progress_callback.get_state_snapshot())
        print(f"Saved current model to {latest_model_path}. Resume later with --resume.")
        return model
    
    final_model_path = Path(output_dir) / "final_model"
    model.save(final_model_path)
    latest_model_path = Path(output_dir) / "latest_model"
    model.save(latest_model_path)
    save_training_state(output_dir, model, total_timesteps, extra_state=progress_callback.get_state_snapshot())
    print(f"\nTraining complete! Model saved to {final_model_path} (latest copy: {latest_model_path})")
    
    return model


def evaluate_model(model, val_dialogues: List[Dict], llm: ChatOpenAI, buffer_size: int = 160, encodings_file: Optional[str] = None, n_episodes: int = 50, use_rate_distortion: bool = False, condensation_model: Optional[str] = None, evaluation_model: Optional[str] = None, impact_model: Optional[str] = None, token_model: Optional[str] = None):
    """Evaluate trained model on validation set."""
    print(f"\nEvaluating model on {n_episodes} episodes...")
    
    env = create_env(val_dialogues, llm, buffer_size=buffer_size, encodings_file=encodings_file, use_rate_distortion=use_rate_distortion, condensation_model=condensation_model, evaluation_model=evaluation_model, impact_model=impact_model, token_model=token_model)
    
    correct = 0
    total = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get('correct', False):
            correct += 1
        total += 1
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Accuracy = {correct/max(total,1)*100:.1f}%")
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nFinal accuracy: {accuracy*100:.1f}% ({correct}/{total})")
    
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FALMA coefficients with PPO")
    parser.add_argument("--dataset", type=str, default="data/dataset_100.json", help="Path to dataset JSON file")
    parser.add_argument("--output", type=str, default="models/falma_ppo", help="Output directory for model")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--buffer-size", type=int, default=160, help="Token buffer size")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/validation split ratio")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model name")
    parser.add_argument("--encodings", type=str, help="Path to pre-computed encodings file (optional)")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--no-subproc", action="store_true", help="Use DummyVecEnv instead of SubprocVecEnv (slower but safer)")
    parser.add_argument("--rate-distortion", action="store_true", help="Use rate-distortion condensation instead of pruning")
    parser.add_argument("--condensation-model", type=str, default=None, help="Model name for condensation (registered with LlmProvider). If not specified, uses --model.")
    parser.add_argument("--evaluation-model", type=str, default=None, help="Model name for evaluation (registered with LlmProvider). If not specified, uses --model for standard pruning.")
    parser.add_argument("--impact-model", type=str, default=None, help="Model name for computing impact factors (registered with LlmProvider). If not specified, uses default remote LLM.")
    parser.add_argument("--token-model", type=str, default=None, help="Model name for token counting (registered with LlmProvider). If not specified, uses default remote LLM.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the most recent checkpoint in the output directory.")
    parser.add_argument("--resume-path", type=str, default=None, help="Explicit checkpoint path (.zip) to resume from.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate for PPO optimiser when starting new training (faster initially).")
    parser.add_argument("--resume-learning-rate", type=float, default=None, help="Override PPO learning rate when resuming from a checkpoint.")
    parser.add_argument("--lr-decay-factor", type=float, default=0.7, help="Multiplicative factor applied to learning rate when plateau detected (set >=1.0 to disable). Default 0.7 for gradual decay.")
    parser.add_argument("--lr-patience", type=int, default=2, help="Number of consecutive evaluation plateaus before applying learning rate decay.")
    parser.add_argument("--lr-threshold", type=float, default=0.5, help="Minimum improvement in validation accuracy (percentage points) to reset plateau counter.")
    parser.add_argument("--min-learning-rate", type=float, default=1e-5, help="Lower bound for automatic learning rate decay.")
    
    args = parser.parse_args()
    
    # If resuming, find the latest run subdirectory
    if args.resume and not args.resume_path:
        output_path = Path(args.output)
        # Look for run subdirectories
        run_dirs = sorted(output_path.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if run_dirs:
            # Use the latest run directory
            args.output = str(run_dirs[0])
            print(f"Resuming from latest run directory: {args.output}")
    
    # If not resuming and output directory exists, create timestamped subdirectory to prevent overwriting
    if not args.resume and not args.resume_path:
        output_path = Path(args.output)
        if output_path.exists() and any(output_path.iterdir()):
            # Create timestamped subdirectory
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            args.output = str(output_path / f"run_{timestamp}")
            print(f"Output directory exists. Creating new run directory: {args.output}")
    
    # Register models based on user requests
    provider = get_provider()
    use_rate_distortion = getattr(args, 'rate_distortion', False)
    condensation_model = getattr(args, 'condensation_model', None)
    evaluation_model = getattr(args, 'evaluation_model', None)
    impact_model = getattr(args, 'impact_model', None)
    token_model = getattr(args, 'token_model', None)
    
    # Register models from command-line arguments
    # Register model specified by --model as default remote
    model_name_short = None
    try:
        gpt_llm = ChatOpenAI(model=args.model, temperature=0, openai_api_key=OPENAI_API_KEY)
        model_name_short = args.model.replace('-', '').replace('_', '').lower()  # e.g., 'gpt-4o' -> 'gpt4o'
        provider.register(model_name_short, Gpt(gpt_llm), default_remote=True)
        print(f"Registered {model_name_short} ({args.model}) as default remote LLM")
    except Exception as e:
        print(f"Warning: Could not register default model: {e}")
    
    # Register condensation model if specified
    if use_rate_distortion:
        if condensation_model is None:
            # Use the default model if not specified
            condensation_model = model_name_short
        
        if condensation_model == 'llama31_8b':
            try:
                llama_llm = TogetherAI(api_key=TOGETHER_API_KEY, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
                provider.register('llama31_8b', llama_llm, default_remote=False)
                print(f"Registered {condensation_model} for condensation")
            except Exception as e:
                print(f"Warning: Could not register {condensation_model}: {e}")
        elif condensation_model and condensation_model != model_name_short:
            # Register custom condensation model if different from default
            try:
                if not provider.is_available(condensation_model):
                    cond_llm = ChatOpenAI(model=args.model, temperature=0, openai_api_key=OPENAI_API_KEY)
                    provider.register(condensation_model, Gpt(cond_llm), default_remote=False)
                    print(f"Registered {condensation_model} for condensation")
            except Exception as e:
                print(f"Warning: Could not register {condensation_model}: {e}")
    
    # Register evaluation model if specified
    if evaluation_model:
        if evaluation_model == 'gpt4o':
            try:
                gpt_llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
                provider.register('gpt4o', Gpt(gpt_llm), default_remote=True)
                print(f"Registered {evaluation_model} for evaluation")
            except Exception as e:
                print(f"Warning: Could not register {evaluation_model}: {e}")
    elif model_name_short:
        # For standard pruning, use the model from --model
        evaluation_model = model_name_short  # Use same as default remote
    
    # Register impact model if specified
    if impact_model:
        if not provider.is_available(impact_model):
            try:
                impact_llm = ChatOpenAI(model=args.model, temperature=0, openai_api_key=OPENAI_API_KEY)
                provider.register(impact_model, Gpt(impact_llm), default_remote=False)
                print(f"Registered {impact_model} for impact factors")
            except Exception as e:
                print(f"Warning: Could not register {impact_model}: {e}")
    
    # Register token model if specified
    if token_model:
        # Try to register if not already registered
        if not provider.is_available(token_model):
            try:
                token_llm = ChatOpenAI(model=args.model, temperature=0, openai_api_key=OPENAI_API_KEY)
                provider.register(token_model, Gpt(token_llm), default_remote=False)
                print(f"Registered {token_model} for token counting")
            except Exception as e:
                print(f"Warning: Could not register {token_model}: {e}")
    
    run_metadata = {
        "dataset": args.dataset,
        "output": args.output,
        "target_timesteps": args.timesteps,
        "buffer_size": args.buffer_size,
        "train_ratio": args.train_ratio,
        "model": args.model,
        "n_envs": args.n_envs,
        "use_subproc": not args.no_subproc,
        "rate_distortion": use_rate_distortion,
        "condensation_model": condensation_model,
        "evaluation_model": evaluation_model,
        "impact_model": impact_model,
        "token_model": token_model,
        "learning_rate": args.learning_rate,
        "resume_learning_rate": args.resume_learning_rate,
        "lr_decay_factor": args.lr_decay_factor,
        "lr_patience": args.lr_patience,
        "lr_threshold": args.lr_threshold,
        "min_learning_rate": args.min_learning_rate,
        "resume": args.resume,
        "resume_path": args.resume_path,
    }
    write_run_metadata(args.output, run_metadata)

    # Load dataset
    dialogues = load_dataset(args.dataset)
    
    train_dialogues, val_dialogues = split_dataset(dialogues, train_ratio=args.train_ratio)
    llm = ChatOpenAI(model=args.model, temperature=0, openai_api_key=OPENAI_API_KEY)
    
    state_path = Path(args.output) / "training_state.json"
    if state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as state_file:
                previous_state = json.load(state_file)
            if "latest_theta_mean" in previous_state and previous_state["latest_theta_mean"] is not None:
                theta_str = ", ".join(f"{float(v):.3f}" for v in previous_state["latest_theta_mean"])
                print(f"Previous theta_mean: [{theta_str}]")
            if "best_val_accuracy" in previous_state and previous_state["best_val_accuracy"] is not None:
                print(f"Previous best validation accuracy: {previous_state['best_val_accuracy']:.1f}%")
            if "current_learning_rate" in previous_state and previous_state["current_learning_rate"] is not None:
                print(f"Previous learning rate: {previous_state['current_learning_rate']:.2e}")
        except Exception as exc:
            print(f"Warning: Could not read training state metadata: {exc}")
    
    resume_checkpoint = args.resume_path
    if args.resume and resume_checkpoint is None:
        resume_checkpoint = find_latest_checkpoint(args.output)
        if resume_checkpoint:
            print(f"Auto-detected resume checkpoint: {resume_checkpoint}")
        else:
            print("No checkpoint found for auto-resume; starting fresh.")
    
    model = train_ppo(
        train_dialogues,
        val_dialogues,
        llm,
        output_dir=args.output,
        total_timesteps=args.timesteps,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        encodings_file=args.encodings,
        n_envs=args.n_envs,
        use_subproc=not args.no_subproc,
        use_rate_distortion=use_rate_distortion,
        condensation_model=condensation_model,
        evaluation_model=evaluation_model,
        impact_model=impact_model,
        token_model=token_model,
        resume_path=resume_checkpoint,
        resume_learning_rate=args.resume_learning_rate,
        lr_decay_factor=args.lr_decay_factor,
        lr_patience=args.lr_patience,
        lr_threshold=args.lr_threshold,
        min_learning_rate=args.min_learning_rate,
    )
    
    accuracy = evaluate_model(model, val_dialogues, llm, buffer_size=args.buffer_size, encodings_file=args.encodings, use_rate_distortion=use_rate_distortion, condensation_model=condensation_model, evaluation_model=evaluation_model, impact_model=impact_model, token_model=token_model)
    
    print(f"\nTraining complete! Final validation accuracy: {accuracy*100:.1f}%")


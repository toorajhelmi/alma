"""Train OFALMA coefficients using PPO."""
import argparse
import json
import os
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import torch.nn as nn
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.ofalma_env import OFALMAEnv
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
    """Create OFALMA environment (encodings will be auto-loaded/computed)."""
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
    env = OFALMAEnv(dialogues, llm, buffer_size=buffer_size, encodings_file=encodings_file, 
                   use_rate_distortion=use_rate_distortion, condensation_model=condensation_model,
                   evaluation_model=evaluation_model, impact_model=impact_model, token_model=token_model)
    return env


def train_ppo(
    train_dialogues: List[Dict],
    val_dialogues: List[Dict],
    llm: ChatOpenAI,
    output_dir: str = "models/ofalma_ppo",
    total_timesteps: int = 10000,
    buffer_size: int = 160,
    learning_rate: float = 3e-4,
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
):
    """Train PPO agent to learn OFALMA coefficients."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute encodings for full dataset once (train + validation)
    all_dialogues = train_dialogues + val_dialogues
    from training.ofalma_env import OFALMAEnv
    temp_env = OFALMAEnv(all_dialogues, llm, buffer_size=buffer_size, encodings_file=encodings_file,
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
    
    from core.ofalma import theta
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
    
    print(f"Using {n_envs} parallel environments (n_steps={n_steps}, collecting {n_steps * n_envs} steps per update)")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=f"{output_dir}/checkpoints",
        name_prefix="ppo_ofalma",
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
        def __init__(self, print_freq: int = 50, eval_callback=None, verbose: int = 1, n_steps_per_update: int = 128):
            super().__init__(verbose)
            self.eval_callback = eval_callback
            self.episode_infos = []  # Store info dicts from completed episodes
            self._last_printed_episode_count = 0
            
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
                            # Print immediately when first episode completes
                            if len(self.episode_infos) == 1:
                                print(f"\n[Step {self.num_timesteps}] First episode completed!")
            
            # Print whenever we have new episodes
            has_new_episodes = len(self.episode_infos) > self._last_printed_episode_count
            
            if len(self.episode_infos) > 0 and has_new_episodes:
                recent_window = min(100, len(self.episode_infos))
                recent_infos = self.episode_infos[-recent_window:]
                
                recent_correct = [1 if info.get('correct', False) else 0 for info in recent_infos]
                train_accuracy = sum(recent_correct) / len(recent_correct) * 100 if recent_correct else 0.0
                total_episodes = len(self.episode_infos)
                
                # Calculate average coefficients and threshold from info dicts
                coeffs_text = ""
                if recent_infos:
                    avg_coeffs = {
                        'theta_S': sum(info.get('theta_S', 0.0) for info in recent_infos) / len(recent_infos),
                        'theta_R': sum(info.get('theta_R', 0.0) for info in recent_infos) / len(recent_infos),
                        'theta_Q': sum(info.get('theta_Q', 0.0) for info in recent_infos) / len(recent_infos),
                        'theta_E': sum(info.get('theta_E', 0.0) for info in recent_infos) / len(recent_infos),
                    }
                    coeffs_text = (f" | θ_S={avg_coeffs['theta_S']:.2f} θ_R={avg_coeffs['theta_R']:.2f} "
                                 f"θ_Q={avg_coeffs['theta_Q']:.2f} θ_E={avg_coeffs['theta_E']:.2f}")
                
                val_text = ""
                if self.eval_callback and hasattr(self.eval_callback, 'last_mean_reward'):
                    try:
                        reward = float(self.eval_callback.last_mean_reward)
                        # Reward is [-1, 1], convert to accuracy [0, 100%]
                        if not (np.isinf(reward) or np.isnan(reward)) and -1 <= reward <= 1:
                            val_acc = (reward + 1) / 2 * 100
                            val_text = f" | Val Accuracy: {val_acc:.1f}%"
                    except (ValueError, TypeError):
                        pass
                
                print(f"\n[Step {self.num_timesteps}] Episodes: {total_episodes} | "
                      f"Train Accuracy: {train_accuracy:.1f}% ({sum(recent_correct)}/{len(recent_correct)}){coeffs_text}{val_text}")
                self._last_printed_episode_count = len(self.episode_infos)
            return True
    
    progress_callback = ProgressWithValidationCallback(
        print_freq=10,  # Not used anymore, kept for compatibility
        eval_callback=eval_callback,
        n_steps_per_update=n_steps * n_envs  # Not used anymore, kept for compatibility
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Train dialogues: {len(train_dialogues)}")
    print(f"Validation dialogues: {len(val_dialogues)}")
    print(f"Each episode = 1 dialogue")
    print(f"Total episodes: ~{total_timesteps // 1}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, progress_callback],
        progress_bar=True,
    )
    
    final_model_path = f"{output_dir}/final_model"
    model.save(final_model_path)
    print(f"\nTraining complete! Model saved to {final_model_path}")
    
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
    parser = argparse.ArgumentParser(description="Train OFALMA coefficients with PPO")
    parser.add_argument("--dataset", type=str, default="data/dataset_100.json", help="Path to dataset JSON file")
    parser.add_argument("--output", type=str, default="models/ofalma_ppo", help="Output directory for model")
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
    
    args = parser.parse_args()
    
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
        # Try to register if not already registered
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
    
    # Load dataset
    dialogues = load_dataset(args.dataset)
    
    train_dialogues, val_dialogues = split_dataset(dialogues, train_ratio=args.train_ratio)
    llm = ChatOpenAI(model=args.model, temperature=0, openai_api_key=OPENAI_API_KEY)
    
    model = train_ppo(
        train_dialogues,
        val_dialogues,
        llm,
        output_dir=args.output,
        total_timesteps=args.timesteps,
        buffer_size=args.buffer_size,
        encodings_file=args.encodings,
        n_envs=args.n_envs,
        use_subproc=not args.no_subproc,
        use_rate_distortion=use_rate_distortion,
        condensation_model=condensation_model,
        evaluation_model=evaluation_model,
        impact_model=impact_model,
        token_model=token_model,
    )
    
    accuracy = evaluate_model(model, val_dialogues, llm, buffer_size=args.buffer_size, encodings_file=args.encodings, use_rate_distortion=use_rate_distortion, condensation_model=condensation_model, evaluation_model=evaluation_model, impact_model=impact_model, token_model=token_model)
    
    print(f"\nTraining complete! Final validation accuracy: {accuracy*100:.1f}%")


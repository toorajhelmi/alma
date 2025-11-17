"""Gymnasium environment for OFALMA RL training."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
import hashlib
from langchain_openai import ChatOpenAI
from core.ofalma import apply_ofalma, apply_ofalma_rate_distortion, llm_impact_factors
from core.experiment_utility import get_token_count_fn
from main import INSTRUCTION, OPENAI_API_KEY, parse_json_response


class OFALMAEnv(gym.Env):
    """Environment for learning OFALMA coefficients via RL."""
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self, 
        dialogues: List[Dict], 
        llm: ChatOpenAI, 
        buffer_size: int = 160,
        encodings: Optional[Dict] = None,
        encodings_file: Optional[str] = None,
        render_mode: Optional[str] = None,
        use_rate_distortion: bool = False,
        condensation_model: Optional[str] = None,
        evaluation_model: Optional[str] = None,
        impact_model: Optional[str] = None,
        token_model: Optional[str] = None
    ):
        """
        Initialize environment.
        
        Args:
            dialogues: List of dialogues with 'dialogue' and 'answer' keys
            llm: Language model instance for OFALMA and evaluation
            buffer_size: Token limit for pruning
            encodings: Pre-computed impact factors (optional, will load from file if None)
            encodings_file: Path to encodings file (auto-generated if None, auto-computed if missing)
            render_mode: Optional render mode
        """
        super().__init__()
        
        self.dialogues = dialogues
        self.llm = llm
        self.buffer_size = buffer_size
        self.render_mode = render_mode
        self.use_rate_distortion = use_rate_distortion
        self.condensation_model = condensation_model
        self.evaluation_model = evaluation_model
        self.impact_model = impact_model
        self.token_model = token_model
        
        if encodings is not None:
            self.encodings = encodings
        elif encodings_file:
            self.encodings = self._load_or_compute_encodings(encodings_file)
        else:
            self.encodings = self._load_or_compute_encodings(None)
        
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        max_facts = max(len(d['dialogue']) - 1 for d in dialogues) if dialogues else 20
        self.max_facts = max_facts
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(max_facts, 4),
            dtype=np.float32
        )
        
        self.current_episode = 0
        self.token_fn = get_token_count_fn(llm)
    
    def _compute_dataset_hash(self, dialogues: List[Dict]) -> str:
        """Compute hash of dataset for versioning."""
        content = json.dumps(
            [(d.get('id'), d.get('dialogue', [])) for d in dialogues],
            sort_keys=True
        )
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_encodings_file_path(self, encodings_file: Optional[str] = None) -> str:
        """Get encodings file path (auto-generate if None, defaults to NPZ format)."""
        if encodings_file:
            return encodings_file
        dataset_hash = self._compute_dataset_hash(self.dialogues)
        return f"data/encodings/dataset_{dataset_hash}.encodings.npz"
    
    def _verify_encodings_match(self, dialogues: List[Dict], encoding_data: Dict) -> bool:
        """Verify that encodings contain all needed dialogues (can be from larger dataset)."""
        encodings = encoding_data.get('encodings', {})
        for dialogue_data in dialogues:
            dialogue_id = dialogue_data.get('id', None)
            if dialogue_id is None:
                continue
            if dialogue_id not in encodings:
                return False
        return True
    
    def _load_encodings(self, encodings_file: str) -> Optional[Dict]:
        """Load pre-computed encodings from file (supports both JSON and NPZ)."""
        try:
            if encodings_file.endswith('.npz'):
                # Load NPZ format
                data = np.load(encodings_file)
                dataset_hash = str(data['dataset_hash'][0])
                total_dialogues = int(data['total_dialogues'][0])
                dataset_source = str(data['dataset_source'][0])
                
                dialogue_ids = data['dialogue_ids']
                impact_factors = data['impact_factors']  # (num_dialogues, max_facts, 3)
                facts_counts = data['facts_counts']
                
                # Convert back to dict format for compatibility
                encodings = {}
                for i, dialogue_id in enumerate(dialogue_ids):
                    facts_count = int(facts_counts[i])
                    factors_array = impact_factors[i, :facts_count, :]  # (facts_count, 3)
                    
                    fact_encodings = []
                    for j in range(facts_count):
                        fact_encodings.append({
                            "S": float(factors_array[j, 0]),
                            "Q": float(factors_array[j, 1]),
                            "E": float(factors_array[j, 2])
                        })
                    
                    encodings[int(dialogue_id)] = {
                        "facts_count": facts_count,
                        "impact_factors": fact_encodings
                    }
                
                return {
                    "metadata": {
                        "dataset_hash": dataset_hash,
                        "total_dialogues": total_dialogues,
                        "dataset_source": dataset_source
                    },
                    "encodings": encodings
                }
            else:
                # Load JSON format (backward compatibility)
                with open(encodings_file, 'r') as f:
                    data = json.load(f)
                return data
        except Exception as e:
            print(f"Error loading encodings from {encodings_file}: {e}")
            return None
    
    def _compute_encodings(self, dialogues: List[Dict], verbose: bool = False) -> Dict:
        """Compute impact factors for all dialogues."""
        encodings = {}
        total = len(dialogues)
        
        if verbose:
            print(f"Computing encodings for {total} dialogues...")
        
        for idx, dialogue_data in enumerate(dialogues, 1):
            dialogue_id = dialogue_data.get('id', idx)
            dialogue = dialogue_data['dialogue']
            
            facts = dialogue[:-1] if len(dialogue) > 1 else dialogue
            
            if verbose and idx % 10 == 0:
                print(f"  Processing {idx}/{total}...")
            
            try:
                impact_factors = llm_impact_factors(facts, verbose=False, impact_model=self.impact_model)
                
                fact_encodings = []
                for fact_idx, fact in enumerate(facts):
                    factors = impact_factors[fact_idx] if fact_idx < len(impact_factors) else {}
                    fact_encodings.append({
                        "S": float(factors.get("S", 0.5)),
                        "Q": float(factors.get("Q", 0.5)),
                        "E": float(factors.get("E", 0.5)),
                    })
                
                encodings[dialogue_id] = {
                    "facts_count": len(facts),
                    "impact_factors": fact_encodings
                }
                
            except Exception as e:
                if verbose:
                    print(f"  Error processing dialogue {dialogue_id}: {e}")
                encodings[dialogue_id] = {
                    "facts_count": len(facts),
                    "impact_factors": [{"S": 0.5, "Q": 0.5, "E": 0.5} for _ in facts]
                }
        
        return encodings
    
    def _save_encodings(self, encodings: Dict, dialogues: List[Dict], encodings_file: str):
        """Save encodings to file with metadata (saves as NPZ format)."""
        dataset_hash = self._compute_dataset_hash(dialogues)
        
        # Determine file extension
        if not encodings_file.endswith('.npz') and not encodings_file.endswith('.json'):
            encodings_file = encodings_file.replace('.json', '.npz') if encodings_file.endswith('.encodings') else encodings_file + '.npz'
        
        if encodings_file.endswith('.npz'):
            # Save as NPZ format
            dialogue_ids = []
            impact_factors_list = []
            facts_counts = []
            
            max_facts = 0
            for dialogue_id in sorted(encodings.keys(), key=int):
                encoding = encodings[dialogue_id]
                facts_count = encoding['facts_count']
                factors = encoding['impact_factors']
                
                dialogue_ids.append(int(dialogue_id))
                facts_counts.append(facts_count)
                
                # Convert [{S, Q, E}, ...] to array
                factors_array = np.array([[f['S'], f['Q'], f['E']] for f in factors], dtype=np.float32)
                impact_factors_list.append(factors_array)
                max_facts = max(max_facts, facts_count)
            
            # Pad all to same size
            padded_factors = []
            for factors in impact_factors_list:
                padded = np.zeros((max_facts, 3), dtype=np.float32)
                padded[:len(factors)] = factors
                padded_factors.append(padded)
            
            # Convert to numpy arrays
            dialogue_ids = np.array(dialogue_ids, dtype=np.int32)
            impact_factors = np.array(padded_factors, dtype=np.float32)
            facts_counts = np.array(facts_counts, dtype=np.int32)
            
            os.makedirs(os.path.dirname(encodings_file) if os.path.dirname(encodings_file) else ".", exist_ok=True)
            
            np.savez_compressed(
                encodings_file,
                dialogue_ids=dialogue_ids,
                impact_factors=impact_factors,
                facts_counts=facts_counts,
                dataset_hash=np.array([dataset_hash], dtype='U20'),
                total_dialogues=np.array([len(dialogues)], dtype=np.int32),
                dataset_source=np.array(['auto-computed'], dtype='U50')
            )
            
            file_size = os.path.getsize(encodings_file) / 1024
            print(f"Encodings saved to {encodings_file} ({file_size:.1f} KB)")
        else:
            # Save as JSON (fallback)
            metadata = {
                "dataset_hash": dataset_hash,
                "total_dialogues": len(dialogues),
                "dataset_source": "auto-computed"
            }
            
            result = {
                "metadata": metadata,
                "encodings": encodings
            }
            
            os.makedirs(os.path.dirname(encodings_file) if os.path.dirname(encodings_file) else ".", exist_ok=True)
            
            with open(encodings_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Encodings saved to {encodings_file}")
    
    def _load_or_compute_encodings(self, encodings_file: Optional[str] = None) -> Dict:
        """
        Load encodings from file, or compute if file doesn't exist or doesn't match.
        
        Args:
            encodings_file: Path to encodings file (auto-generated if None)
        
        Returns:
            Dict of encodings: {dialogue_id: {facts_count, impact_factors: [...]}}
        """
        if encodings_file is None:
            encodings_file = self._get_encodings_file_path(None)
        
        if os.path.exists(encodings_file):
            encoding_data = self._load_encodings(encodings_file)
            if encoding_data and self._verify_encodings_match(self.dialogues, encoding_data):
                print(f"✓ Loaded pre-computed encodings from {encodings_file}")
                return encoding_data.get('encodings', {})
            else:
                print(f"Encodings file exists but doesn't contain all needed dialogues, recomputing...")
        
        print(f"Computing encodings for {len(self.dialogues)} dialogues...")
        print(f"This may take a few minutes. Results will be saved to {encodings_file}")
        encodings = self._compute_encodings(self.dialogues, verbose=True)
        self._save_encodings(encodings, self.dialogues, encodings_file)
        
        return encodings
    
    def _encode_dialogue(self, dialogue: List[str], dialogue_id: Optional[int] = None) -> np.ndarray:
        """Encode dialogue using OFALMA impact factors."""
        facts = dialogue[:-1] if len(dialogue) > 1 else dialogue
        total_facts = len(facts)
        
        if total_facts == 0:
            return np.zeros((self.max_facts, 4), dtype=np.float32)
        impact_factors_list = None
        if dialogue_id is not None and dialogue_id in self.encodings:
            encoding_data = self.encodings[dialogue_id]
            if encoding_data.get('facts_count') == total_facts:
                impact_factors_list = encoding_data.get('impact_factors', [])
        
        if impact_factors_list is None:
            from core.ofalma import llm_impact_factors
            computed = llm_impact_factors(facts, verbose=False, impact_model=self.impact_model)
            impact_factors_list = [
                {
                    "S": float(f.get("S", 0.5)),
                    "Q": float(f.get("Q", 0.5)),
                    "E": float(f.get("E", 0.5))
                }
                for f in computed
            ]
        
        state_matrix = []
        for idx, fact in enumerate(facts):
            factors = impact_factors_list[idx] if idx < len(impact_factors_list) else {}
            S = float(factors.get("S", 0.5))
            Q = float(factors.get("Q", 0.5))
            E = float(factors.get("E", 0.5))
            R = (idx + 1) / total_facts
            state_matrix.append([S, Q, E, R])
        
        state = np.array(state_matrix, dtype=np.float32)
        if len(state) < self.max_facts:
            padding = np.zeros((self.max_facts - len(state), 4), dtype=np.float32)
            state = np.vstack([state, padding])
        elif len(state) > self.max_facts:
            state = state[:self.max_facts]
        
        return state
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to a new dialogue. Uses deterministic cycling if seed is provided."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            # Deterministic: cycle through dialogues based on episode count
            if not hasattr(self, '_episode_counter'):
                self._episode_counter = 0
            self.current_episode = self._episode_counter % len(self.dialogues)
            self._episode_counter += 1
        else:
            # Random sampling for training
            self.current_episode = np.random.randint(0, len(self.dialogues))
        
        dialogue_data = self.dialogues[self.current_episode]
        self.current_dialogue = dialogue_data['dialogue']
        self.correct_answer = dialogue_data['answer']
        self.current_dialogue_id = dialogue_data.get('id', self.current_episode)
        
        self.current_facts = self.current_dialogue[:-1] if len(self.current_dialogue) > 1 else self.current_dialogue
        self.current_question = self.current_dialogue[-1] if len(self.current_dialogue) > 1 else ""
        
        state = self._encode_dialogue(self.current_dialogue, dialogue_id=self.current_dialogue_id)
        
        info = {
            'dialogue_id': dialogue_data.get('id', self.current_episode),
            'problem_type': dialogue_data.get('problem_type', 'unknown'),
            'correct_answer': self.correct_answer
        }
        
        return state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Handle action if it's from vec_env (might be 2D or need flattening)
        if isinstance(action, np.ndarray):
            if action.ndim > 1:
                action = action.flatten()
            if len(action) != 4:
                raise ValueError(f"Expected action of length 4, got {len(action)}")
        
        theta_S, theta_R, theta_Q, theta_E = action[0], action[1], action[2], action[3]
        
        from core.ofalma import theta as global_theta
        import core.ofalma as ofalma
        
        original_theta = global_theta.copy()
        ofalma.theta = {
            "S": float(theta_S),
            "R": float(theta_R),
            "Q": float(theta_Q),
            "E": float(theta_E)
        }
        
        try:
            impact_factors_list = None
            if self.current_dialogue_id is not None and self.current_dialogue_id in self.encodings:
                encoding_data = self.encodings[self.current_dialogue_id]
                if encoding_data.get('facts_count') == len(self.current_facts):
                    impact_factors_list = encoding_data.get('impact_factors', [])
            
            if self.use_rate_distortion:
                # Use rate-distortion condensation
                condensation_model = getattr(self, 'condensation_model', None)
                impact_model = getattr(self, 'impact_model', None)
                token_model = getattr(self, 'token_model', None)
                try:
                    condensed_facts, stats = apply_ofalma_rate_distortion(
                        self.current_facts,
                        self.buffer_size,
                        impact_factors_list=impact_factors_list,
                        k=3.0,  # Match default k value
                        condensation_model=condensation_model,
                        impact_model=impact_model,
                        token_model=token_model
                    )
                    kept_facts = condensed_facts
                    removed_facts = []  # Rate-distortion condenses, doesn't remove
                except Exception as e:
                    # Fallback to standard pruning if rate-distortion fails
                    print(f"Rate-distortion failed, falling back to standard pruning: {e}")
                    kept_facts, removed_facts, stats = apply_ofalma(
                        self.current_facts, 
                        self.buffer_size,
                        impact_factors_list=impact_factors_list,
                        impact_model=impact_model,
                        token_model=token_model
                    )
            else:
                # Use standard OFALMA pruning
                impact_model = getattr(self, 'impact_model', None)
                token_model = getattr(self, 'token_model', None)
            kept_facts, removed_facts, stats = apply_ofalma(
                self.current_facts, 
                    self.buffer_size,
                    impact_factors_list=impact_factors_list,
                    impact_model=impact_model,
                    token_model=token_model
            )
            
            prompt_parts = []
            if kept_facts:
                prompt_parts.append("\n".join(kept_facts))
            if self.current_question:
                prompt_parts.append(self.current_question)
            if INSTRUCTION:
                prompt_parts.append(INSTRUCTION)
            final_prompt = "\n".join(prompt_parts)
            
            from llm import GetLlm
            # Use specified evaluation model or default GPT
            if self.evaluation_model:
                try:
                    llm = GetLlm(llm=self.evaluation_model, fallback=False)
                    if llm is None:
                        raise ValueError(f"LLM '{self.evaluation_model}' not found")
                except Exception as e:
                    print(f"ERROR: Failed to get evaluation model '{self.evaluation_model}': {e}")
                    raise
            else:
                # For standard pruning, use specified evaluation model or default
                if self.evaluation_model:
                    llm = GetLlm(llm=self.evaluation_model, fallback=False)
                else:
                    llm = GetLlm(fallback=True)  # Use default remote LLM
                if llm is None:
                    raise ValueError("Failed to get LLM. Make sure a model is registered.")
            
            raw_content = llm.send(final_prompt)
            parsed = parse_json_response(raw_content)
            
            # Handle both dict and int returns from parse_json_response
            if isinstance(parsed, dict):
                gpt_answer = parsed.get("answer", 0)
            elif isinstance(parsed, int):
                gpt_answer = parsed
            else:
                gpt_answer = 0
            
            # Ensure both are integers for comparison
            try:
                gpt_answer = int(gpt_answer) if gpt_answer is not None else None
            except (ValueError, TypeError):
                gpt_answer = None
            
            try:
                if self.correct_answer is None:
                    correct_answer = None
                else:
                    correct_answer = int(self.correct_answer)
            except (ValueError, TypeError):
                correct_answer = None
            
            # Only mark as correct if both answers are valid and match
            if gpt_answer is None or correct_answer is None:
                is_correct = False
            else:
                is_correct = gpt_answer == correct_answer
            
            reward = 1.0 if is_correct else -1.0
            terminated = True
            truncated = False
            
            info = {
                'gpt_answer': gpt_answer,
                'correct_answer': correct_answer,
                'correct': is_correct,
                'kept_facts': len(kept_facts),
                'removed_facts': len(removed_facts),
                'total_facts': len(self.current_facts),
                'theta_S': float(theta_S),
                'theta_R': float(theta_R),
                'theta_Q': float(theta_Q),
                'theta_E': float(theta_E),
                'all_facts_kept': len(kept_facts) == len(self.current_facts),
                'pruned_ratio': len(removed_facts) / max(len(self.current_facts), 1),
            }
            
        except Exception as e:
            reward = -1.0
            terminated = True
            truncated = False
            info = {
                'error': str(e),
                'correct': False,
                'gpt_answer': None,
                'correct_answer': self.correct_answer if hasattr(self, 'correct_answer') else None,
                'kept_facts': 0,
                'removed_facts': 0,
                'total_facts': len(self.current_facts) if hasattr(self, 'current_facts') else 0,
                'theta_S': float(theta_S),
                'theta_R': float(theta_R),
                'theta_Q': float(theta_Q),
                'theta_E': float(theta_E),
            }
            # Print error to help debug issues
            print(f"ERROR in step(): {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            ofalma.theta = original_theta
        
        state = self._encode_dialogue(self.current_dialogue, dialogue_id=self.current_dialogue_id)
        
        return state, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            print(f"Episode: {self.current_episode}")
            print(f"Facts: {len(self.current_facts)}")
            print(f"Correct answer: {self.correct_answer}")


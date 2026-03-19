"""
PPO Alignment Script for Malware Analysts
------------------------------------------
This script uses Proximal Policy Optimization (PPO) to align a model with
analyst preferences. It requires:
1. A Policy Model (the one we train)
2. A Reference Model (usually the SFT baseline)
3. A Reward mechanism (Scoring function or Reward Model)
"""

import sys
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Mock vLLM for Mac/Import checks (TRL 0.14.0+)
try:
    import vllm
except ImportError:
    from types import ModuleType
    mock_vllm = ModuleType("vllm")
    mock_vllm.distributed = ModuleType("vllm.distributed")
    mock_vllm.distributed.device_communicators = ModuleType("vllm.distributed.device_communicators")
    mock_vllm.LLM = type("LLM", (), {})
    mock_vllm.SamplingParams = type("SamplingParams", (), {})
    sys.modules["vllm"] = mock_vllm
    sys.modules["vllm.distributed"] = mock_vllm.distributed
    sys.modules["vllm.distributed.device_communicators"] = mock_vllm.distributed.device_communicators

# Robust TRL & Transformers Imports
try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model
except ImportError:
    try:
        from trl.models import AutoModelForCausalLMWithValueHead
        from trl.trainer import PPOConfig, PPOTrainer
        from trl.models.utils import create_reference_model
    except ImportError:
        # Final fallback for some specific TRL versions
        from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
        from trl.trainer.ppo_config import PPOConfig
        from trl.trainer.ppo_trainer import PPOTrainer
        from trl.models import create_reference_model

# Optional Unsloth for CUDA speedup
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

# Monkey-patch trl.trainer.ppo_trainer to handle AutoModelForCausalLMWithValueHead 
# (Direct patch because PPOTrainer uses "from .utils import ...")
import trl.trainer.ppo_trainer as ppo_trainer_module
from transformers.utils import ModelOutput

# 1. Configuration
config_args = {
    "learning_rate": 1.41e-5,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_mini_batches": 1,
    "response_length": 128,
    "total_episodes": 100,
    "kl_coef": 0.05,
    "cliprange": 0.2,
    "vf_coef": 0.1,
    "output_dir": "research/results/ppo_malware",
}

# 2. Custom Reward Logic (The 'Analyst Preference')
def calculate_reward(query, response):
    """
    In a real scenario, this would be a Reward Model (RM).
    Here, we reward professional tone and technical precision.
    """
    reward = 0.0
    # Reward for professional keywords
    keywords = ["persistence", "obfuscation", "entropy", "mutex", "api call"]
    for kw in keywords:
        if kw in response.lower():
            reward += 0.2
            
    # Penalty for being too wordy or generic
    if len(response.split()) > 300:
        reward -= 0.5
        
    return reward

def patched_get_reward(model, query_responses, pad_token_id, context_length):
    """Replacement for trl.trainer.utils.get_reward that handles both values and rewards."""
    # 1. Compute sequence lengths (needed for both)
    from trl.trainer.utils import first_true_indices
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    
    # 2. Compute Values (from value head of value_model)
    with torch.no_grad():
        out = model(query_responses)
        full_value = out[2] # shape (batch, seq)

    # 3. Compute Custom Rewards (from python function)
    queries = tokenizer.batch_decode(query_responses[:, :context_length], skip_special_tokens=True)
    responses = tokenizer.batch_decode(query_responses[:, context_length:], skip_special_tokens=True)
    rewards = [calculate_reward(q, r) for q, r in zip(queries, responses)]
    rewards_tensor = torch.tensor(rewards, device=query_responses.device, dtype=torch.float32)
    
    return (full_value, rewards_tensor, sequence_lengths)

ppo_trainer_module.get_reward = patched_get_reward

# Also patch forward to wrap tuples in ModelOutput
old_forward = ppo_trainer_module.forward
def patched_forward(*args, **kwargs):
    out = old_forward(*args, **kwargs)
    if isinstance(out, tuple):
        if len(out) == 2:
            # PolicyAndValueWrapper.forward returns (policy_out, value_logits)
            policy_out, value_logits = out
            if isinstance(policy_out, tuple):
                policy_out = ModelOutput(logits=policy_out[0], loss=policy_out[1])
            return policy_out, value_logits
        elif len(out) >= 3:
            # AutoModelForCausalLMWithValueHead.forward returns (logits, loss, value)
            return ModelOutput(logits=out[0], loss=out[1])
    return out
ppo_trainer_module.forward = patched_forward

# 3. Load Model
device = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available(): device = "cuda"

config = PPOConfig(
    exp_name="malware_ppo",
    **config_args
)

if HAS_UNSLOTH and device == "cuda":
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        load_in_4bit = True,
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
else:
    # Fallback to standard Transformers if no CUDA or Unsloth
    print(f"Using standard Transformers on {device}")
    model_id = "unsloth/Llama-3.2-1B-Instruct" if device == "mps" else "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
    if device != "cuda": model = model.to(device)

# Ensure generation_config and base_model_prefix are available (needed for trl 0.13+)
if not hasattr(model, "generation_config"):
    model.generation_config = model.pretrained_model.generation_config
if not hasattr(model, "base_model_prefix"):
    model.base_model_prefix = "pretrained_model"

# Attach v_head as score method (needed for PolicyAndValueWrapper)
if not hasattr(model, "score"):
    model.score = model.v_head

ref_model = create_reference_model(model)

# 4. Data Preparation
base_dir = os.getcwd()
dataset_path = os.path.join(base_dir, "data/finetuning/malware_sft_data.jsonl")
dataset = load_dataset("json", data_files=dataset_path, split="train")

# New PPO expects prompts in a specific format if using Chat templates, but here we keep it simple
def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["input"], truncation=True, max_length=512)
    return sample

dataset = dataset.map(tokenize, batched=False, remove_columns=dataset.column_names)
dataset.set_format(type="torch")

# 5. Trainer
# Since we monkey-patched get_reward, reward_model is not really used for computation, 
# but it must be passed as an nn.Module. We pass the model itself as a placeholder.
ppo_trainer = PPOTrainer(
    args=config,
    model=model,
    ref_model=ref_model,
    processing_class=tokenizer,
    train_dataset=dataset,
    reward_model=model, 
    value_model=model,
)

# 6. Training Loop
print("Starting PPO Alignment...")
# In 0.13.0, we just call train()
ppo_trainer.train()

# 7. Save
save_path = os.path.join(base_dir, "research/results/malware_analyst_ppo")
ppo_trainer.save_model(save_path)
print(f"PPO Model saved to {save_path}")

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

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model

# Optional Unsloth for CUDA speedup
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

# 1. Configuration
config_args = {
    "learning_rate": 1.41e-5,
    "batch_size": 4,
    "mini_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "kl_coef": 0.05,
    "cliprange": 0.2,
}

# Add model_name only if supported (older TRL versions)
try:
    config = PPOConfig(model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit", **config_args)
except TypeError:
    config = PPOConfig(**config_args)

# 2. Load Model
device = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available(): device = "cuda"

if HAS_UNSLOTH and device == "cuda":
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.model_name,
        load_in_4bit = True,
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
else:
    # Fallback to standard Transformers if no CUDA or Unsloth
    print(f"Using standard Transformers on {device}")
    model_id = "unsloth/Llama-3.2-1B-Instruct" if device == "mps" else config.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
    if device != "cuda": model = model.to(device)
ref_model = create_reference_model(model)

# 3. Custom Reward Logic (The 'Analyst Preference')
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
        
    return torch.tensor(reward)

# 4. Data Preparation
base_dir = os.getcwd()
dataset_path = os.path.join(base_dir, "data/finetuning/malware_sft_data.jsonl")
dataset = load_dataset("json", data_files=dataset_path, split="train")

def tokenize(sample):
    # Prepare query for PPO
    sample["input_ids"] = tokenizer.encode(sample["input"])
    sample["query"] = sample["input"]
    return sample

dataset = dataset.map(tokenize, batched=False)
dataset.set_format(type="torch")

# 5. Trainer
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
)

# 6. Training Loop
print("Starting PPO Alignment...")
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 128,
}

for epoch in range(1):
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        # Get response from policy
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # Compute rewards
        rewards = [calculate_reward(q, r) for q, r in zip(batch["query"], batch["response"])]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

# 7. Save
save_path = os.path.join(base_dir, "malware_analyst_ppo")
model.save_pretrained(save_path)
print(f"PPO Model saved to {save_path}")

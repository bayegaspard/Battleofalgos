"""
PPO Alignment Script for Malware Analysts
------------------------------------------
This script uses Proximal Policy Optimization (PPO) to align a model with
analyst preferences. It requires:
1. A Policy Model (AutoModelForCausalLM - the one we train)
2. A Reference Model (a frozen copy of policy)
3. A Value Model (AutoModelForSequenceClassification - has built-in .score head)
4. A Reward mechanism (Custom Python scoring function)

Compatible with trl >= 0.13.0 (new Trainer-based PPOTrainer).
Does NOT require AutoModelForCausalLMWithValueHead.
"""

import os
import sys
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from trl import PPOConfig, PPOTrainer
from trl.trainer.utils import first_true_indices
import trl.trainer.ppo_trainer as ppo_trainer_module

# -----------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"Using device: {device}")

base_dir = os.getcwd()

# Determine model to use (smaller on MPS to avoid OOM)
if device == "mps":
    MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
    VALUE_MODEL_ID = MODEL_ID  # same arch, no quant needed on MPS
else:
    MODEL_ID = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    # Value model MUST be non-quantized: normal_() init fails on uint8/4-bit tensors.
    # Using 1B here to save VRAM; it shares the same tokenizer & architecture.
    VALUE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"

PPO_CONFIG = PPOConfig(
    exp_name="malware_ppo",
    learning_rate=1.41e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_mini_batches=1,
    response_length=128,
    total_episodes=100,
    kl_coef=0.05,
    cliprange=0.2,
    vf_coef=0.1,
    num_sample_generations=0,  # Disabled: no eval_dataset provided
    output_dir=os.path.join(base_dir, "research/results/ppo_malware"),
)

# -----------------------------------------------------------------------
# 2. Custom Reward Logic (The 'Analyst Preference')
# -----------------------------------------------------------------------
def calculate_reward(query: str, response: str) -> float:
    """
    Rule-based proxy reward function.
    Rewards professional malware-analysis keywords and penalizes verbosity.
    In production, replace with a trained Reward Model.
    """
    reward = 0.0
    keywords = ["persistence", "obfuscation", "entropy", "mutex", "api call"]
    for kw in keywords:
        if kw in response.lower():
            reward += 0.2
    if len(response.split()) > 300:
        reward -= 0.5
    return reward


def make_patched_get_reward(tok, value_model):
    """
    Factory that returns a get_reward replacement closure.

    The new PPOTrainer calls get_reward twice per episode:
     - First call (with value_model): fetches full per-token values from critic.
     - Second call (with reward_model): fetches final scalar rewards.

    Because we use the same function for both, we detect which is being called
    by inspecting whether the model has a `.score` method (value_model does).
    """
    def patched_get_reward(model, query_responses, pad_token_id, context_length):
        seq_lens = (
            first_true_indices(query_responses[:, context_length:] == pad_token_id)
            - 1
            + context_length
        )

        # --- Value call: model has .score, return per-token value estimates ---
        if hasattr(model, "score"):
            attention_mask = query_responses != pad_token_id
            position_ids = attention_mask.cumsum(1) - attention_mask.long()
            input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
            with torch.no_grad():
                out = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                    output_hidden_states=True,
                )
                # .score is a Linear(hidden_size, 1) → shape (B, T, 1)
                full_value = model.score(out.hidden_states[-1]).squeeze(-1)
            return (full_value, None, seq_lens)

        # --- Reward call: model is policy (no .score), return scalar rewards ---
        queries   = tok.batch_decode(query_responses[:, :context_length], skip_special_tokens=True)
        responses = tok.batch_decode(query_responses[:, context_length:],  skip_special_tokens=True)
        rewards   = [calculate_reward(q, r) for q, r in zip(queries, responses)]
        rewards_t = torch.tensor(rewards, device=query_responses.device, dtype=torch.float32)
        return (None, rewards_t, seq_lens)

    return patched_get_reward


# -----------------------------------------------------------------------
# 3. Load Models & Tokenizer
# -----------------------------------------------------------------------
print(f"Loading tokenizer and models from: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # PPO needs left-padding for generation

# Policy model (the one we train)
# Use bfloat16 on CUDA: more numerically stable than fp16 (no NaN/inf after
# a few gradient steps), and H100 has native bf16 hardware support.
if device == "cuda":
    dtype = torch.bfloat16
elif device == "mps":
    dtype = torch.float32  # MPS doesn't fully support fp16/bf16
else:
    dtype = torch.float32

policy_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
policy_model = policy_model.to(device)

# Reference model (frozen copy of policy)
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
ref_model = ref_model.to(device)
for p in ref_model.parameters():
    p.requires_grad_(False)

# Value model: AutoModelForSequenceClassification has a linear .score head
# num_labels=1 → scalar critic output.
# NOTE: Must load from a NON-quantized checkpoint. Quantized (4-bit / uint8)
#       weights cause "normal_kernel_cuda not implemented for Byte" when
#       PyTorch tries to randomly init the new score head via normal_().
value_model = AutoModelForSequenceClassification.from_pretrained(
    VALUE_MODEL_ID, num_labels=1, torch_dtype=dtype, ignore_mismatched_sizes=True
)
value_model = value_model.to(device)

# Inject our monkey-patched get_reward AFTER models are built
ppo_trainer_module.get_reward = make_patched_get_reward(tokenizer, value_model)

# -----------------------------------------------------------------------
# 4. Data Preparation
# -----------------------------------------------------------------------
dataset_path = os.path.join(base_dir, "data/finetuning/malware_sft_data.jsonl")
dataset = load_dataset("json", data_files=dataset_path, split="train")

def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(
        sample["input"], truncation=True, max_length=512
    )
    return sample

dataset = dataset.map(tokenize, batched=False, remove_columns=dataset.column_names)
dataset.set_format(type="torch")

# -----------------------------------------------------------------------
# 5. PPOTrainer Setup
# -----------------------------------------------------------------------
ppo_trainer = PPOTrainer(
    args=PPO_CONFIG,
    model=policy_model,
    ref_model=ref_model,
    processing_class=tokenizer,
    train_dataset=dataset,
    reward_model=policy_model,  # placeholder (logic overridden by monkey-patch)
    value_model=value_model,
)

# -----------------------------------------------------------------------
# 6. Train
# -----------------------------------------------------------------------
print("Starting PPO Alignment...")
ppo_trainer.train()
# -----------------------------------------------------------------------
# 7. Save
# -----------------------------------------------------------------------
save_path = os.path.join(base_dir, "research/results/malware_analyst_ppo")
os.makedirs(save_path, exist_ok=True)

# ppo_trainer.save_model() crashes for 4-bit quantized base models because
# save_pretrained() calls revert_weight_conversion() which raises NotImplementedError.
# Instead, we unwrap the model and save the raw state dict directly.
try:
    unwrapped_policy = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).policy
    state_dict = {k: v.cpu() for k, v in unwrapped_policy.state_dict().items()}
    torch.save(state_dict, os.path.join(save_path, "policy_state_dict.pt"))
    tokenizer.save_pretrained(save_path)
    print(f"PPO Policy state dict saved to {save_path}/policy_state_dict.pt")
    print("To reload: model.load_state_dict(torch.load('policy_state_dict.pt'))")
except Exception as e:
    # Fallback: save directly from the original policy_model handle
    print(f"accelerator unwrap failed ({e}), saving directly from policy_model...")
    state_dict = {k: v.cpu() for k, v in policy_model.state_dict().items()}
    torch.save(state_dict, os.path.join(save_path, "policy_state_dict.pt"))
    tokenizer.save_pretrained(save_path)
    print(f"PPO Policy state dict saved to {save_path}/policy_state_dict.pt")

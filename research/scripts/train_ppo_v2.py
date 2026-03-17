"""
PPO Alignment Script for Malware Analysts
------------------------------------------
This script uses Proximal Policy Optimization (PPO) to align a model with
analyst preferences. It requires:
1. A Policy Model (the one we train)
2. A Reference Model (usually the SFT baseline)
3. A Reward mechanism (Scoring function or Reward Model)
"""

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model
from unsloth import FastLanguageModel

# 1. Configuration
config = PPOConfig(
    model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    learning_rate=1.41e-5,
    batch_size=4,
    mini_batch_size=1,
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
    early_stopping=True,
    target_kl=0.1,
)

# 2. Load Model (with Value Head for PPO)
# Note: Unsloth doesn't natively support ValueHead in the same way as SFT, 
# so we use traditional TRL with Unsloth's optimized weights.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config.model_name,
    load_in_4bit = True,
)

# PPO needs a 'Value Head' to estimate rewards
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
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
dataset = load_dataset("json", data_files="data/finetuning/malware_sft_data.jsonl", split="train")

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
model.save_pretrained("malware_analyst_ppo")
print("PPO Model saved to malware_analyst_ppo")

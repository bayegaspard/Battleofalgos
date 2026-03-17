"""
GRPO Training Script for Malware Reasoning
------------------------------------------
This script uses Group Relative Policy Optimization (GRPO) to train a model
to reason through malware reports. It rewards:
1. XML Format (thinking/answer tags)
2. Accuracy (matching ground truth)
3. Reasoning quality (length and density)
"""

import re
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

# 1. Patch Unsloth for RL
PatchFastRL()

# 2. Configuration
model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit" # Good balance of speed/memory
max_seq_length = 1024 # Buffer for thinking
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
    fast_inference = True,
    max_lora_rank = 32,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. Reward Functions

def format_reward_func(completions, **kwargs) -> list[float]:
    """Rewards models for using <thought> and <answer> tags."""
    # Pattern: <thought>...</thought><answer>...</answer>
    pattern = r"^<thought>.*?</thought>\s*<answer>.*?</answer>$"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

def accuracy_reward_func(completions, answer, **kwargs) -> list[float]:
    """Rewards models for matching the ground truth answer."""
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for r, ground_truth in zip(responses, answer):
        # Extract content between <answer> tags
        match = re.search(r"<answer>(.*?)</answer>", r, re.DOTALL)
        if match:
            predicted = match.group(1).strip().lower()
            actual = ground_truth.strip().lower()
            rewards.append(1.0 if actual in predicted else 0.0)
        else:
            rewards.append(0.0)
    return rewards

def reasoning_length_reward_func(completions, **kwargs) -> list[float]:
    """Rewards models for longer 'thinking' sections (basic proxy for density)."""
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for r in responses:
        match = re.search(r"<thought>(.*?)</thought>", r, re.DOTALL)
        if match:
            length = len(match.group(1).split())
            # Reward up to 0.5 for 50+ words of reasoning
            rewards.append(min(length / 100.0, 0.5))
        else:
            rewards.append(0.0)
    return rewards

# 4. Data Preparation
# We need prompt/answer pairs
base_dir = os.getcwd()
dataset_path = os.path.join(base_dir, "data/finetuning/malware_sft_data.jsonl")
raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

def prep_grpo_data(example):
    # Extract the question and ground truth from our SFT data
    # Instruction: "Analyze... and answer..."
    # Input: "REPORT: ... QUESTION: ... OPTIONS: ..."
    # Output: "REASONING: ... CORRECT ANSWERS: ..."
    
    # We strip the reasoning Gemini provided to let the model learn its own
    ground_truth = example["output"].split("CORRECT ANSWERS:")[1].strip()
    
    return {
        "prompt": [
            {"role": "system", "content": "You are a professional malware analyst. Always think step-by-step inside <thought> tags and provide your final selection inside <answer> tags."},
            {"role": "user", "content": example["input"]}
        ],
        "answer": ground_truth
    }

dataset = raw_dataset.map(prep_grpo_data)

# 5. Trainer Configuration
training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_generations = 4, # Group size for GRPO
    max_prompt_length = 512,
    max_completion_length = 512,
    num_train_epochs = 1,
    save_steps = 10,
    max_grad_norm = 0.1,
    output_dir = os.path.join(base_dir, "outputs/grpo"),
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        format_reward_func,
        accuracy_reward_func,
        reasoning_length_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)

# 6. Run Training
print("Starting GRPO Training...")
trainer.train()

# 7. Save
save_path = os.path.join(base_dir, "malware_analyst_grpo")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"GRPO Model saved to {save_path}")

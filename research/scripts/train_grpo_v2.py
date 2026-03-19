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
import os
import sys
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------
# vLLM Compatibility Layer
# -----------------------------------------------------------------------
# On NVIDIA servers with a partial/broken vLLM install the module may land in
# sys.modules with __spec__ = None.  importlib.util.find_spec() raises
# ValueError in that case, which crashes trl on import.  Remove it first so
# trl gets a clean availability check before we install our own mock.
import importlib.machinery

def _clean_broken_vllm():
    """Remove any sys.modules entries whose __spec__ is None."""
    keys = [k for k in list(sys.modules) if k == "vllm" or k.startswith("vllm.")]
    for k in keys:
        if getattr(sys.modules[k], "__spec__", None) is None:
            del sys.modules[k]

_clean_broken_vllm()

# Now import trl (needs a clean sys.modules view of vllm)
try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    try:
        from trl.trainer import GRPOConfig, GRPOTrainer
    except ImportError:
        print("ERROR: TRL GRPO library structure is unexpected. Please run 'pip install trl==0.14.0'")
        raise

# After trl is imported and cached, install a safe vllm mock for anything
# downstream that might try to import it.  We set __spec__ so that future
# find_spec() calls don't raise ValueError.
try:
    import vllm  # use real vllm if genuinely available
except (ImportError, Exception):
    from types import ModuleType
    def _make_mock(name):
        m = ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        return m
    mock_vllm = _make_mock("vllm")
    mock_vllm.distributed = _make_mock("vllm.distributed")
    mock_vllm.distributed.device_communicators = _make_mock("vllm.distributed.device_communicators")
    mock_vllm.LLM = type("LLM", (), {})
    mock_vllm.SamplingParams = type("SamplingParams", (), {})
    sys.modules["vllm"] = mock_vllm
    sys.modules["vllm.distributed"] = mock_vllm.distributed
    sys.modules["vllm.distributed.device_communicators"] = mock_vllm.distributed.device_communicators


# Optional Unsloth for CUDA speedup
try:
    from unsloth import FastLanguageModel, PatchFastRL
    from unsloth import is_bfloat16_supported
    HAS_UNSLOTH = True
    PatchFastRL()
except ImportError:
    HAS_UNSLOTH = False

# Configuration
model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
max_seq_length = 1024
load_in_4bit = True

def is_bfloat16_supported_fallback():
    if HAS_UNSLOTH: return is_bfloat16_supported()
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# 2. Load Model
device = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available(): device = "cuda"

if HAS_UNSLOTH and device == "cuda":
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
else:
    # Fallback to standard Transformers/PEFT
    print(f"Using standard Transformers on {device}")
    from peft import LoraConfig, get_peft_model
    model_id = "unsloth/Llama-3.2-1B-Instruct" if device == "mps" else model_name
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device != "cuda": model = model.to(device)
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

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
    bf16 = False, # MPS doesn't support bf16 fully in all operations
    fp16 = False, # Use fp32 for stability on MPS if needed, or stick to default
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_generations = 2, # Reduced for Mac memory
    max_prompt_length = 256,
    max_completion_length = 256,
    num_train_epochs = 1,
    save_steps = 10,
    max_grad_norm = 0.1,
    output_dir = os.path.join(base_dir, "outputs/grpo"),
    use_vllm = False, # Explicitly disable on Mac
)

# Fix for TRL 0.14.0+ where GRPOTrainer expects warnings_issued dict
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

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

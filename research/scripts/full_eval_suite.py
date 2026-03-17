"""
Comparative Benchmarking Suite
-----------------------------
Evaluates SFT vs GRPO vs PPO models against the benchmark questions.
Usage: python3 scripts/compare_training_methods.py
"""

import os
import json
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
import pandas as pd

# Add project root to path
import sys
sys.path.append(os.getcwd())
from src.evaluation import evaluate

# 1. Configuration
MODELS = {
    "Baseline": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "SFT": "malware_analyst_sft", # Path to saved adapter
    "GRPO": "malware_analyst_grpo",
    "PPO": "malware_analyst_ppo"
}

QUESTIONS_PATH = "data/questions/questions.json"

def run_bench(model_name, adapter_path=None):
    print(f"\n>>> Benchmarking {model_name}...")
    
    if not os.path.exists(adapter_path or model_name) and adapter_path:
        print(f"Skipping {model_name}, adapter not found.")
        return None

    # Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path if adapter_path else model_name,
        load_in_4bit = True,
        max_seq_length = 2048,
    )
    FastLanguageModel.for_inference(model)

    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    predictions = []
    answers = []

    for q in tqdm(questions[:20]): # Test on 20 samples for speed
        prompt = f"### Instruction: Analyze the malware report and answer the question.\n\n### Input: {q['question']}\n\n### Response:"
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Simple extraction logic
        # Expecting something like "Correct Options: A, B"
        try:
            pred = [opt.strip() for opt in response.split("Response:")[1].split() if opt.strip() in ['A', 'B', 'C', 'D']]
            predictions.append(pred)
        except:
            predictions.append([])
            
        answers.append(q.get("correct_options", []))

    results = evaluate(predictions, answers)
    return results

def main():
    final_stats = []
    
    for name, path in MODELS.items():
        # Check if it's the baseline or an adapter
        is_baseline = (name == "Baseline")
        res = run_bench(name, None if is_baseline else path)
        if res:
            res["Model"] = name
            final_stats.append(res)

    # 3. Save & Display
    if final_stats:
        df = pd.DataFrame(final_stats)
        print("\n" + "="*40)
        print("FINAL COMPARATIVE RESULTS")
        print("="*40)
        print(df.to_string(index=False))
        
        df.to_csv("data/benchmark_comparison.csv", index=False)
        print(f"\nDetailed results saved to data/benchmark_comparison.csv")

if __name__ == "__main__":
    main()

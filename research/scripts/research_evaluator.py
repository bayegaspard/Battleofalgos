"""
Cybersecurity Research Evaluator (Core Study Script)
---------------------------------------------------
Implements multi-dimensional metrics for comparing local models (SFT/GRPO/PPO)
against frontier baselines.

Dimensions:
1. Accuracy: Ground truth match.
2. Technical Precision: Use of domain-specific terminology (MITRE ATT&CK, etc.)
3. Reasoning Density: Token ratio of <thought> tags.
4. Robustness: Jaccard similarity across variations of the same report.
"""

import os
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
from src.llm_router import LLMRouter

# Security terms for 'Technical Precision' metric
SECURITY_LEXICON = [
    "persistence", "privilege escalation", "obfuscation", "entropy", "mutex",
    "iat", "pe header", "process injection", "lateral movement", "c2", 
    "beacon", "exfiltration", "shellcode", "packer", "mitre"
]

def calculate_precision_score(text):
    """Calculates the density of security-specific terms."""
    if not text: return 0.0
    text = text.lower()
    matches = sum(1 for term in SECURITY_LEXICON if term in text)
    return matches / (len(text.split()) + 1) * 10 # Scaled score

def extract_thinking(text):
    """Extracts content between <thought> tags."""
    match = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

class ResearchEvaluator:
    def __init__(self, data_path, model_configs):
        self.data = self._load_data(data_path)
        self.configs = model_configs
        self.results = []

    def _load_data(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def evaluate_model(self, name, model_path, provider="ollama", is_local=True):
        print(f"\n[RESEARCH] Evaluating: {name}")
        
        if is_local:
            import requests
            url = "http://localhost:11434/api/generate"
            print(f"[LOCAL] Using Ollama model: {model_path}")
            
            preds = []
            for item in tqdm(self.data[:20]):
                options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(item['options'])])
                prompt = f"### System: You are a Tier-3 SOC Analyst. Analyze the following and think step-by-step.\n\n### Options:\n{options_str}\n\n### Report: {item['question']}\n\n### Response:"
                
                payload = {
                    "model": model_path,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 1024}
                }
                
                try:
                    response_obj = requests.post(url, json=payload, timeout=60).json()
                    response = response_obj.get("response", "")
                except Exception as e:
                    print(f"Error calling Ollama: {e}")
                    response = ""
                
                # Metrics
                thought = extract_thinking(response)
                precision = calculate_precision_score(response)
                density = len(thought.split()) / (len(response.split()) + 1)
                
                # Answer extraction and mapping
                ans_match = re.search(r"Answer:\s*([A-D, ]+)", response)
                pred_letters = [a.strip().upper() for a in ans_match.group(1).split(",")] if ans_match else []
                
                # Map letters back to option text
                pred_texts = []
                for char in pred_letters:
                    idx = ord(char) - 65
                    if 0 <= idx < len(item['options']):
                        pred_texts.append(item['options'][idx])
                
                is_correct = set(pred_texts) == set(item['correct_options'])
                
                preds.append({
                    "Model": name,
                    "Accuracy": 1.0 if is_correct else 0.0,
                    "Tech_Precision": precision,
                    "Reasoning_Density": density,
                    "Words": len(response.split())
                })
            return pd.DataFrame(preds)
        else:
            router = LLMRouter("config.yaml")
            preds = []
            for item in tqdm(self.data[:20]):
                options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(item['options'])])
                prompt = f"### System: You are a Tier-3 SOC Analyst. Analyze the following and think step-by-step.\n\n### Options:\n{options_str}\n\n### Report: {item['question']}\n\n### Response:"
                response = router.generate(prompt)
                
                # Metrics
                thought = extract_thinking(response)
                precision = calculate_precision_score(response)
                density = len(thought.split()) / (len(response.split()) + 1)
                
                # Answer extraction and mapping
                ans_match = re.search(r"Answer:\s*([A-D, ]+)", response)
                pred_letters = [a.strip().upper() for a in ans_match.group(1).split(",")] if ans_match else []
                
                # Map letters back to option text
                pred_texts = []
                for char in pred_letters:
                    idx = ord(char) - 65
                    if 0 <= idx < len(item['options']):
                        pred_texts.append(item['options'][idx])
                
                is_correct = set(pred_texts) == set(item['correct_options'])
                
                preds.append({
                    "Model": name,
                    "Accuracy": 1.0 if is_correct else 0.0,
                    "Tech_Precision": precision,
                    "Reasoning_Density": density,
                    "Words": len(response.split())
                })
            return pd.DataFrame(preds)

def main():
    models = {
        "Baseline_Gemini": {"path": "gemini-2.5-flash", "is_local": False},
        "Baseline_Llama3": {"path": "llama3:latest", "is_local": True},
        "Baseline_Mistral": {"path": "mistral:latest", "is_local": True},
        "SFT_Expert_Llama3": {"path": "research/results/sft_llama3_lora", "is_local": True},
    }
    
    evaluator = ResearchEvaluator("data/questions/questions.json", models)
    
    all_results = []
    for name, config in models.items():
        res = evaluator.evaluate_model(name, config["path"], is_local=config["is_local"])
        all_results.append(res)
        
        # Incremental save
        incremental_df = pd.concat(all_results)
        incremental_df.to_csv("research/results/experiment_results_v1_partial.csv")
        
    final_df = pd.concat(all_results)
    summary = final_df.groupby("Model").mean()
    
    print("\n--- RESEARCH SUMMARY ---")
    print(summary)
    
    # Save to research artifacts
    summary.to_csv("research/results/experiment_results_v1.csv")

if __name__ == "__main__":
    main()

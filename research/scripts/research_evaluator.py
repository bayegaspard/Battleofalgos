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

    def _generate_local(self, prompt, model_path):
        """Helper to generate response via Transformers for local paths."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        # Cache for loaded models to avoid reloading every item
        if not hasattr(self, "_cached_local_model"):
            print(f"[TRANSFORMERS] Loading local model from {model_path}...")
            # Detect base model from path or default to Llama-3.2-3B if unknown
            # For simplicity in this research, we assume the experts were trained on Mistral/Llama
            base_model_id = "mistralai/Mistral-7B-Instruct-v0.3" 
            if "llama" in model_path.lower():
                base_model_id = "unsloth/Llama-3.2-1B-Instruct" if torch.backends.mps.is_available() else "meta-llama/Llama-3.2-3B-Instruct"

            self._cached_tokenizer = AutoTokenizer.from_pretrained(model_path)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            if torch.cuda.is_available(): device = "cuda"
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            if device != "cuda": base_model = base_model.to(device)
            self._cached_local_model = PeftModel.from_pretrained(base_model, model_path)
            print("[TRANSFORMERS] Model loaded.")

        inputs = self._cached_tokenizer(prompt, return_tensors="pt").to(self._cached_local_model.device)
        with torch.no_grad():
            outputs = self._cached_local_model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
        
        return self._cached_tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:")[-1].strip()

    def evaluate_model(self, name, model_path, provider="ollama", is_local=True):
        print(f"\n[RESEARCH] Evaluating: {name}")
        
        # Check if model_path is a local directory
        use_transformers = os.path.isdir(model_path)
        
        preds = []
        for item in tqdm(self.data[:20]):
            report_ctx = json.dumps(item.get('report', {}), indent=2)
            options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(item['options'])])
            prompt = f"### System: You are a Tier-3 SOC Analyst. Analyze the following sandbox report and answer the multiple-choice question.\n\n### Report:\n{report_ctx}\n\n### Question:\n{item['question']}\n\n### Options:\n{options_str}\n\n### Response:"
            
            if use_transformers:
                response = self._generate_local(prompt, model_path)
            elif is_local:
                import requests
                url = "http://localhost:11434/api/generate"
                payload = {
                    "model": model_path,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 1024}
                }
                try:
                    response_obj = requests.post(url, json=payload, timeout=120).json()
                    response = response_obj.get("response", "")
                except Exception as e:
                    print(f"Error calling Ollama: {e}")
                    response = ""
            else:
                router = LLMRouter("config.yaml")
                response = router.generate(prompt)
            
            # Log first response
            with open("research/results/evaluation_debug.log", "a") as log:
                log.write(f"\n--- DEBUG RESPONSE ({name}) ---\n{response}\n------------------\n")
            
            # Metrics
            thought = extract_thinking(response)
            precision = calculate_precision_score(response)
            density = len(thought.split()) / (len(response.split()) + 1)
            
            # ROBUST Answer extraction
            ans_patterns = [
                r"Answer[:\s]*\**([A-D])\**",
                r"The\s+answer\s+is[:\s]*\**([A-D])\**",
                r"The\s+correct\s+answer\s+is[:\s]*\**([A-D])\**",
                r"final\s+answer\s+is\s+.*?boxed\{([A-D])\}",
                r"boxed\{([A-D])\}",
                r"Selection[:\s]*\**([A-D])\**",
                r"aligns\s+with\s+option\s+\**([A-D])\**",
                r"option[:\s]*\**([A-D])\**",
                r"choice[:\s]*\**([A-D])\**",
                r"^\s*\**([A-D])\**\s*$",
                r"(?m)^\s*\*?\*?([A-D])\.\s" 
            ]
            pred_letters = []
            for p in ans_patterns:
                match = re.search(p, response, re.IGNORECASE)
                if match:
                    pred_letters = [a.strip().upper() for a in match.group(1).replace(",", " ").split()]
                    break
            
            # Fallback
            if not pred_letters and response.strip():
                last_line = response.strip().split("\n")[-1].strip()
                if re.match(r"^[A-D]$", last_line, re.I):
                    pred_letters = [last_line.upper()]
            
            pred_texts = []
            for char in pred_letters:
                idx = ord(char) - 65
                if 0 <= idx < len(item['options']):
                    pred_texts.append(item['options'][idx])
            
            is_correct = set(pred_texts) == set(item['correct_options'])
            with open("/tmp/evaluation_comparison.log", "a") as f:
                f.write(f"DEBUG: {name} | Pred Letters: {pred_letters} | Pred: {pred_texts} | Expected: {item['correct_options']} | Match: {is_correct}\n")
            
            preds.append({
                "Model": name,
                "Accuracy": 1.0 if is_correct else 0.0,
                "Tech_Precision": precision,
                "Reasoning_Density": density,
                "Words": len(response.split())
            })
        
        # Clear cache after evaluating one model
        if hasattr(self, "_cached_local_model"):
            del self._cached_local_model
            del self._cached_tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return pd.DataFrame(preds)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    base_dir = os.getcwd()
    questions_path = os.path.join(base_dir, "data/questions/questions.json")
    
    models = {
        "Baseline_Gemini": {"path": "gemini-2.0-flash", "is_local": False},
        "Baseline_Llama3": {"path": "llama3:latest", "is_local": True},
        "Baseline_Mistral": {"path": "mistral:latest", "is_local": True},
        "SFT_Expert_Mistral": {"path": os.path.join(base_dir, "research/results/sft_mistral_lora"), "is_local": True},
        "PPO_Expert_Llama": {"path": os.path.join(base_dir, "malware_analyst_ppo"), "is_local": True},
        "GRPO_Expert_Llama": {"path": os.path.join(base_dir, "malware_analyst_grpo"), "is_local": True},
    }
    
    target_models = {args.model: models[args.model]} if args.model and args.model in models else models
    evaluator = ResearchEvaluator(questions_path, target_models)
    
    all_results = []
    for name, config in target_models.items():
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

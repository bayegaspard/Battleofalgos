#!/bin/bash

# Battle of Algos: Research Pipeline Automator
# -------------------------------------------

echo "===================================================="
echo "   BATTLE OF ALGOS: RESEARCH PIPELINE AUTOMATOR"
echo "===================================================="

# 1. Environment & VENV Setup
echo "[1/5] Setting up Virtual Environment and Dependencies..."

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install dependencies from requirements.txt
echo "Installing required Python libraries from requirements.txt..."
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r requirements.txt

# Verify installation of critical modules
python3 -c "from google import genai; import peft; print('Dependencies verified.')" || { echo "Dependency verification failed."; exit 1; }

set -e  # Exit on first error

# 2. Dataset Generation (Distillation)
echo -e "\n[2/6] Regenerating questions and distilling silver-standard reasoning..."
python3 scripts/generate_questions.py
python3 research/scripts/distill_reasoning.py

# 3. Model Training (SFT -> PPO -> GRPO)
echo -e "\n[3/6] Starting Multi-Stage Training Pipeline..."

echo "Stage A: SFT (Reasoning Distillation)..."
python3 research/scripts/train_sft.py

echo "Stage B: PPO Alignment..."
python3 research/scripts/train_ppo_v2.py

echo "Stage C: GRPO (Rule-Based Reinforcement)..."
python3 research/scripts/train_grpo_v2.py

# 4. Comprehensive Evaluation
echo -e "\n[4/6] Running Multi-Model Benchmarking (Baselines vs. Experts)..."

# Ensure expert models were actually saved
if [ ! -d "research/results/sft_mistral_lora" ]; then
    echo "CRITICAL ERROR: SFT Model directory not found. Training likely failed."
    exit 1
fi

# Evaluate each model individually to avoid OOM or specific failures blocking the whole run
python3 research/scripts/research_evaluator.py --model Baseline_Gemini
python3 research/scripts/research_evaluator.py --model Baseline_Llama3
python3 research/scripts/research_evaluator.py --model Baseline_Mistral
python3 research/scripts/research_evaluator.py --model SFT_Expert_Mistral
python3 research/scripts/research_evaluator.py --model PPO_Expert_Llama
python3 research/scripts/research_evaluator.py --model GRPO_Expert_Llama

# 5. Visualization & Paper Updates
echo -e "\n[5/6] Generating IEEE-style Research Visualizations..."
python3 research/scripts/visualize_results.py

echo -e "\n===================================================="
echo "   RESEARCH PIPELINE COMPLETE"
echo "   Results saved in: research/results/"
echo "===================================================="

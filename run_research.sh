#!/bin/bash

# Battle of Algos: End-to-End Research Pipeline
# Author: Senior Cybersecurity AI Research Scientist

set -e

echo "===================================================="
echo "   BATTLE OF ALGOS: RESEARCH PIPELINE AUTOMATOR"
echo "===================================================="

# 1. Environment Check
echo "[1/5] Checking Environment and Models..."
if command -v nvidia-smi &> /dev/null; then
    echo "Found NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Running on MacOS (MPS support available)"
else
    echo "No GPU detected. Training might be extremely slow."
fi

# Ensure Ollama models are pulled
if command -v ollama &> /dev/null; then
    echo "Pulling Ollama models for baseline..."
    ollama pull mistral:latest
    ollama pull llama3:latest
else
    echo "Ollama not found. Please install it for baseline evaluations."
fi

# 2. Baseline Evaluation
echo -e "\n[2/5] Running Multi-Model Baseline Evaluation..."
python3 research/scripts/research_evaluator.py

# 3. Supervised Fine-Tuning (SFT)
echo -e "\n[3/5] Starting SFT Training (Reasoning Distillation)..."
# Note: Ensure data/finetuning/malware_sft_data.jsonl exists or run distillation first
if [ ! -f "data/finetuning/malware_sft_data.jsonl" ]; then
    echo "SFT Data not found! Run distillation script first."
    # python3 scripts/distill_reasoning.py --samples 50
fi
python3 research/scripts/train_sft.py

# 4. Comparative Evaluation
echo -e "\n[4/5] Running Evaluation on Fine-Tuned Models..."
# We will update research_evaluator.py to include the new SFT checkpoint path
# For now, we rerun the evaluator which will pick up the current state
python3 research/scripts/research_evaluator.py

# 5. Visualization & Paper Artifacts
echo -e "\n[5/5] Generating IEEE-style Visualizations..."
python3 research/scripts/visualize_results.py

echo -e "\n===================================================="
echo "   PROCESS COMPLETE: CHECK research/results/ "
echo "===================================================="

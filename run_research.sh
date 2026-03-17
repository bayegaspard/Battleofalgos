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

# Regenerate questions with context
echo "Regenerating questions with report context..."
python3 scripts/generate_questions.py

# 2. Baseline Evaluation
echo -e "\n[2/5] Running Multi-Model Baseline Evaluation..."
python3 research/scripts/research_evaluator.py

# 3. SFT Training
echo -e "\n[3/5] Starting SFT Training (Reasoning Distillation)..."
python3 research/scripts/train_sft.py

# 4. Expert Evaluation
echo -e "\n[4/5] Running Expert vs. Baseline Comparison..."
python3 research/scripts/research_evaluator.py

# 5. Visualization
echo -e "\n[5/5] Generating IEEE-style Research Visualizations..."
python3 research/scripts/visualize_results.py

echo -e "\n===================================================="
echo "   RESEARCH PIPELINE COMPLETE"
echo "   Results saved in: research/results/"
echo "===================================================="

from src.llm_router import LLMRouter
from src.evaluation import evaluate
import json
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

def run():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found.")
        return

    llm = LLMRouter(config_path)

    questions_path = "data/questions/questions.json"
    if not os.path.exists(questions_path):
        print(f"Questions file {questions_path} not found. Run generate_questions.py first.")
        return

    with open(questions_path, "r") as f:
        questions = json.load(f)

    predictions = []
    answers = []

    for q in questions:
        # The prompt for the benchmark is the generated question itself
        prompt = q.get("question", "")
        if not prompt:
            continue

        print(f"Evaluating: {prompt[:50]}...")
        response = llm.generate(prompt)
        
        # In a real benchmark, you might need to parse the response to extract the selected options
        # For now, we assume the response is the selected option(s) in a parsable format
        try:
            # Basic parsing if LLM returns "Option A, B"
            if isinstance(response, str):
                # This is a simplification; real evaluation would use regex to find options
                pred_options = [opt.strip() for opt in response.replace(",", " ").split() if len(opt.strip()) > 0]
                predictions.append(pred_options)
            else:
                predictions.append(response)
        except:
            predictions.append([])
            
        answers.append(q.get("correct_options", []))

    result = evaluate(predictions, answers)
    print("\nBenchmark Results:")
    print(f"Accuracy: {result['accuracy']:.2f}")
    print(f"Jaccard: {result['jaccard']:.2f}")

if __name__ == "__main__":
    run()

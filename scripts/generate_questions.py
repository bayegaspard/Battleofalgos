import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.llm_router import LLMRouter
from src.malware_question_gen import MalwareQuestionGenerator
from src.report_filter import filter_report
import json

def generate():
    llm = LLMRouter("config.yaml")

    generator = MalwareQuestionGenerator(
        llm,
        "prompts/malware_question.jinja"
    )

    # Updated to point to existing hybrid-analysis data
    reports_dir = "data/hybrid-analysis"
    output_path = "data/questions/questions.json"

    questions = []

    if not os.path.exists(reports_dir):
        print(f"Directory {reports_dir} does not exist.")
        return

    for root, dirs, files in os.walk(reports_dir, followlinks=True):
        for f in files:
            if f == "questions.json" or f.startswith("."):
                continue
                
            file_path = os.path.join(root, f)
            with open(file_path, "r") as report_file:
                try:
                    report = json.load(report_file)
                except json.JSONDecodeError:
                    continue

            # Filter report to reduce tokens
            filtered_report = filter_report(report)

            print(f"Generating question for {f}...")
            q = generator.generate(filtered_report)

            if q:
                questions.append(q)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2)
    
    print(f"Generated {len(questions)} questions saved to {output_path}")

if __name__ == "__main__":
    generate()

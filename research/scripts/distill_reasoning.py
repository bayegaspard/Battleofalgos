import os
import sys
import json
from jinja2 import Template

# Add project root to path
sys.path.append(os.getcwd())

from src.llm_router import LLMRouter
from src.report_filter import filter_report

class MalwareDataDistiller:
    def __init__(self, llm, template_path):
        self.llm = llm
        with open(template_path, "r") as f:
            self.template = Template(f.read())

    def distill(self, report):
        prompt = self.template.render(report=json.dumps(report))
        response = self.llm.generate(prompt)

        try:
            # Attempt to find JSON block if LLM adds preamble
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response:
                response = "{" + response.split("{", 1)[1].rsplit("}", 1)[0] + "}"
                
            return json.loads(response)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None

def main():
    base_dir = os.getcwd()
    reports_dir = os.path.join(base_dir, "data/hybrid-analysis")
    output_path = os.path.join(base_dir, "data/finetuning/malware_sft_data.jsonl")
    config_path = os.path.join(base_dir, "config.yaml")
    prompt_path = os.path.join(base_dir, "prompts/malware_distillation.jinja")

    llm = LLMRouter(config_path)
    distiller = MalwareDataDistiller(llm, prompt_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(reports_dir):
        print(f"Directory {reports_dir} does not exist.")
        return

    # To save tokens and time during test, we might want to limit the number of reports
    processed_count = 0
    
    with open(output_path, "w") as out_file:
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

                filtered_report = filter_report(report)
                print(f"Distilling reasoning for {f}...")
                
                result = distiller.distill(filtered_report)
                
                if result:
                    # Format for SFT: Instruction, Input, Output
                    # We can use a ChatML style or Alpaca style
                    sft_entry = {
                        "instruction": "Analyze the following malware sandbox report and answer the multiple-choice question. Provide your reasoning before the final answer.",
                        "input": f"REPORT:\n{json.dumps(filtered_report)}\n\nQUESTION: {result['question']}\nOPTIONS: {', '.join(result['options'])}",
                        "output": f"REASONING: {result['rationale']}\n\nCORRECT ANSWERS: {', '.join(result['correct_options'])}"
                    }
                    out_file.write(json.dumps(sft_entry) + "\n")
                    processed_count += 1
                    
                    if processed_count >= 50: # Increased for research study
                         print("Reached limit for distillation.")
                         break
            if processed_count >= 50:
                break

    if processed_count == 0:
        print("ERROR: No malware reports found or processed. Check data/hybrid-analysis/")
        sys.exit(1)

    print(f"Successfully distilled {processed_count} samples to {output_path}")

if __name__ == "__main__":
    main()

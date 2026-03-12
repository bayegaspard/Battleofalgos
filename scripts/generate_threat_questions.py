from src.llm_router import LLMRouter
from src.threat_question_gen import ThreatQuestionGenerator
from src.data_loader import extract_pdf_pages
import json
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

def generate():
    llm = LLMRouter("config.yaml")

    generator = ThreatQuestionGenerator(
        llm,
        "prompts/threat_category.jinja"
    )

    reports_dir = "data/threat_reports"
    output_path = "data/questions/threat_questions.json"

    categories = ["Malware Analysis", "Threat Actor Attribution", "Campaign Analysis", "Indicator of Compromise"]

    questions = []

    if not os.path.exists(reports_dir):
        print(f"Directory {reports_dir} does not exist.")
        return

    for f in os.listdir(reports_dir):
        if not f.endswith(".pdf"):
            continue
            
        file_path = os.path.join(reports_dir, f)
        print(f"Processing PDF: {f}...")
        
        pages = extract_pdf_pages(file_path)
        if not pages:
            continue
            
        # Combine pages for simplicity or process per page
        full_text = "\n".join(pages)
        # Truncate to avoid token limits if necessary
        content = full_text[:10000] 

        for category in categories:
            print(f"Generating question for category: {category}...")
            # We reuse the generator but inject category into context if needed
            # For simplicity, we'll assume the template handles the category
            q = generator.generate(content, category=category)

            if q:
                q["category"] = category
                q["source"] = f
                questions.append(q)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2)
    
    print(f"Generated {len(questions)} threat intel questions saved to {output_path}")

if __name__ == "__main__":
    generate()

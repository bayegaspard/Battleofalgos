import json
import os

def load_json_report(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_all_json_reports(directory_path):
    reports = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".json"):
                reports.append(load_json_report(os.path.join(root, file)))
    return reports

def save_benchmark_questions(questions, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2)

def extract_pdf_pages(file_path):
    """Simple PDF page extraction. Requires pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return [page.extract_text() for page in reader.pages]
    except ImportError:
        print("pypdf not installed. Please run: pip install pypdf")
        return []
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return []

def load_threat_reports(directory_path):
    reports = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".pdf"):
                reports.append(os.path.join(root, file))
    return reports

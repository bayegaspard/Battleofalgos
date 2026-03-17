"""
Cybersecurity LLM Metrics Library (v2)
--------------------------------------
Formalized metrics for the IEEE research paper "Battle of Algos".
"""

import re
import math

def calculate_reasoning_density(response_text):
    """
    Reasoning Density (RD):
    RD = (Thinking Tokens) / (Total Tokens)
    where Thinking Tokens are enclosed in <thought> tags.
    """
    thought_match = re.search(r"<thought>(.*?)</thought>", response_text, re.DOTALL)
    if not thought_match:
        return 0.0
    
    thought_tokens = len(thought_match.group(1).split())
    total_tokens = len(response_text.split())
    
    return thought_tokens / total_tokens if total_tokens > 0 else 0.0

def calculate_technical_precision(response_text):
    """
    Technical Precision (TP):
    TP = (Count of Sector Lexicon Terms) / (Total Words)
    Lexicon is based on SANS, MITRE ATT&CK, and ISO 27001 terminology.
    """
    lexicon = [
        "persistence", "obfuscation", "lateral movement", "privilege escalation",
        "command and control", "beaconing", "regsvr32", "powershell",
        "mimikatz", "cobalt strike", "living off the land", "lotl",
        "mitre", "att&ck", "indicator of compromise", "ioc"
    ]
    
    count = 0
    clean_text = response_text.lower()
    for term in lexicon:
        count += len(re.findall(f"\\b{re.escape(term)}\\b", clean_text))
    
    total_words = len(response_text.split())
    
    # Scale density for readability (x100 for percentage-like score)
    return (count / total_words) * 100 if total_words > 0 else 0.0

def calculate_confidence_score(accuracy, reasoning_density):
    """
    Confidence Score (CS):
    A weighted aggregation of accuracy and the model's 'internal confidence' 
    proxied by reasoning density.
    CS = (Acc * 0.7) + (min(RD, 0.5) * 0.6)
    """
    # Clip RD at 0.5 to prevent over-explaining from inflating the score
    clipped_rd = min(reasoning_density, 0.5)
    return (accuracy * 0.7) + (clipped_rd * 0.6)

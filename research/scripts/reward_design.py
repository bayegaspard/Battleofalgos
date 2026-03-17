"""
Cybersecurity Reward Design (Research Methodology)
-----------------------------------------------
Formalizes the reward functions for GRPO/PPO training in a cybersecurity context.
These functions are designed to reinforce 'Think-before-Action' and 
'Technical Precision' in local models.
"""

import re

def accuracy_reward(predicted_ans, ground_truth):
    """
    Accuracy Reward: Binary 1.0/0.0
    Encourages the model to correctly identify the threat attributes.
    """
    return 1.0 if set(predicted_ans) == set(ground_truth) else 0.0

def reasoning_coherence_reward(thought_text):
    """
    Coherence Reward: Continuous [0.0, 1.0]
    Encourages longer, more structured reasoning.
    Weights: 
    - Length (up to 0.4)
    - Structural markers e.g. 'Therefore', 'Evidence' (up to 0.6)
    """
    reward = 0.0
    words = thought_text.split()
    
    # Word count (capped at 100 words)
    reward += min(len(words) / 100.0, 0.4)
    
    # Structural Markers
    markers = ["evidence", "signature", "persistence", "registry", "network", "therefore"]
    count_markers = sum(1 for m in markers if m in thought_text.lower())
    reward += min(count_markers * 0.1, 0.6)
    
    return reward

def format_adherence_reward(response_text):
    """
    Format Reward: Binary 0.5/0.0
    Strictly rewards the presence of <thought> and <answer> tags.
    """
    pattern = r"^<thought>.*?</thought>\s*<answer>.*?</answer>$"
    return 0.5 if re.match(pattern, response_text, re.DOTALL) else 0.0

def calculate_total_reward(response, ground_truth):
    """
    Combined Reward Function for GRPO:
    Total = Format + Accuracy + Reasoning
    Total range: [0.0, 2.5]
    """
    # 1. Format
    f_reward = format_adherence_reward(response)
    
    # 2. Accuracy
    # Note: Accuracy requires parsing the answer from the tags first
    ans_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    predicted_ans = [a.strip() for a in ans_match.group(1).split(",")] if ans_match else []
    a_reward = accuracy_reward(predicted_ans, ground_truth)
    
    # 3. Reasoning
    thought_match = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL)
    thought_text = thought_match.group(1) if thought_match else ""
    r_reward = reasoning_coherence_reward(thought_text)
    
    return f_reward + a_reward + r_reward

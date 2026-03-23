"""
build_hf_dataset.py
===================
Builds a 1-million-sample HuggingFace dataset of malware analysis
reasoning chains for the RL alignment pipeline (SFT -> PPO -> GRPO).

Each example has:
  - "prompt"        : system + user message (ready for tokenisation)
  - "thought"       : <thought>...</thought> reasoning chain
  - "answer"        : <answer>...</answer> label string
  - "full_response" : thought + answer concatenated (SFT target)
  - metadata        : malware_family, ttps, difficulty, source, augmentation

Strategy to reach 1 M samples
──────────────────────────────
1. SEED generation  (~10 k real Hybrid-Analysis/VirusTotal reports -> Gemini)
2. AUGMENTATION (x100 per seed):
     a. Option-order shuffle          (x4)
     b. Question lexical rephrase     (x3)
     c. Difficulty re-label           (x2)
     d. TTP-perspective prefix        (x5)
     e. Negation-focus variant        (x1)
     f. Random recombination rounds   (fills remaining gap)
3. SHA-256 deduplication
4. Upload to HuggingFace Hub as Parquet (train / validation / test splits)

Usage
─────
  # Generate seeds (needs GEMINI_API_KEY)
  python build_hf_dataset.py --mode seed \\
      --reports data/hybrid-analysis --out data/hf_dataset

  # Augment seeds -> 1 M
  python build_hf_dataset.py --mode augment \\
      --seeds data/hf_dataset/seeds.jsonl \\
      --out data/hf_dataset --target 1000000

  # Push to HuggingFace Hub (needs HF_TOKEN or huggingface-cli login)
  python build_hf_dataset.py --mode push \\
      --dataset data/hf_dataset/dataset_1m.jsonl \\
      --repo valix-ai/malware-reasoning-1m

  # All-in-one
  python build_hf_dataset.py --mode all \\
      --reports data/hybrid-analysis \\
      --repo valix-ai/malware-reasoning-1m
"""

import os
import sys
import json
import random
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Iterator

import google.generativeai as genai
from datasets import Dataset, DatasetDict, Features, Value
from tqdm import tqdm

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Prompts / constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Tier-3 SOC analyst specialising in malware analysis. "
    "Analyse the sandbox report provided and answer the question. "
    "First reason step-by-step inside <thought>...</thought> tags, "
    "citing specific report evidence and MITRE ATT&CK TTPs. "
    "Then state your final answer inside <answer>...</answer> tags."
)

DISTILLATION_PROMPT = """\
You are an expert Malware Researcher and SOC Tier-3 Analyst.

Using the sandbox report below, produce ONE high-quality training sample.

<REPORT>
{report}
</REPORT>

### Task
1. Identify key behaviours: persistence, evasion, C2, lateral movement, exfiltration.
2. Write a 4-option multiple-choice question that tests malware reasoning.
3. Identify the correct option(s).
4. Write a step-by-step rationale that cites specific report fields.
5. Format a complete model response using <thought> and <answer> XML tags.

Return raw JSON (no markdown fences):
{{
  "question":       "...",
  "options":        ["Option A", "Option B", "Option C", "Option D"],
  "correct_options":["Option A"],
  "rationale":      "Detailed step-by-step reasoning citing report evidence...",
  "malware_family": "<Extract from report, e.g. Trickbot or Unknown>",
  "ttps":           ["<Extract TTP 1>", "<Extract TTP 2>"],
  "thought":        "<thought>\\nStep 1: ...\\nStep 2: ...\\n</thought>",
  "answer":         "<answer>Option A</answer>"
}}
"""

HF_FEATURES = Features({
    "id":             Value("string"),
    "prompt":         Value("string"),
    "thought":        Value("string"),
    "answer":         Value("string"),
    "full_response":  Value("string"),
    "question":       Value("string"),
    "options":        Value("string"),        # JSON-encoded list
    "correct_options":Value("string"),        # JSON-encoded list
    "malware_family": Value("string"),
    "ttps":           Value("string"),        # JSON-encoded list
    "difficulty":     Value("string"),
    "source":         Value("string"),
    "augmentation":   Value("string"),
})

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

TTP_PERSPECTIVES = [
    "network behaviour",
    "persistence mechanisms",
    "process injection",
    "file system activity",
    "evasion techniques",
]

PARAPHRASE_MAP = {
    "indicates":   ["exhibits", "demonstrates", "reveals", "shows"],
    "suggests":    ["implies", "points to", "highlights"],
    "most likely": ["primarily", "chiefly", "principally"],
    "Which":       ["What"],
    "identify":    ["determine", "classify", "detect"],
}


# ---------------------------------------------------------------------------
#  Gemini client
# ---------------------------------------------------------------------------

def get_gemini_client(model: str = "gemini-2.5-flash") -> genai.GenerativeModel:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set the GEMINI_API_KEY environment variable before running."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model)


def distill_report(client: genai.GenerativeModel, report: dict) -> dict | None:
    """Call Gemini to generate a reasoning chain for one report."""
    prompt = DISTILLATION_PROMPT.format(report=json.dumps(report, indent=2))
    try:
        resp = client.generate_content(prompt)
        raw = resp.text.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif raw.startswith("```"):
            raw = raw.strip("`").strip()
        return json.loads(raw)
    except Exception as exc:
        log.warning("Gemini distillation error: %s", exc)
        return None


# ---------------------------------------------------------------------------
#  Report loader / filter
# ---------------------------------------------------------------------------

def iter_reports(reports_dir: str) -> Iterator[dict]:
    for path in Path(reports_dir).rglob("*.json"):
        if path.name in {"questions.json"} or path.name.startswith("."):
            continue
        try:
            with open(path) as f:
                yield json.load(f)
        except Exception:
            continue


def filter_report(report: dict) -> dict:
    """Keep only the fields relevant to malware analysis."""
    keys = [
        "file_name", "verdict", "vx_family", "threat_score",
        "processes", "network", "signatures", "file_metadata",
        "registry", "mitre_attcks", "name", "type", "behavior"
    ]
    return {k: report[k] for k in keys if k in report}


# ---------------------------------------------------------------------------
#  Seed generation
# ---------------------------------------------------------------------------

def generate_seeds(
    reports_dir: str,
    out_dir: str,
    limit: int = 10_000,
) -> str:
    """Generate seed examples from real reports using Gemini."""
    out_path = Path(out_dir) / "seeds.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_gemini_client()
    count = 0

    with open(out_path, "w") as fout:
        for report in tqdm(iter_reports(reports_dir), desc="Generating seeds"):
            if count >= limit:
                break
            filtered = filter_report(report)
            result = distill_report(client, filtered)
            if not result:
                continue
            seed = _build_row(
                result=result,
                report=filtered,
                source="hybrid-analysis",
                difficulty=_infer_difficulty(result),
                augmentation="seed",
            )
            fout.write(json.dumps(seed) + "\n")
            count += 1

    log.info("Seed generation complete: %d examples -> %s", count, out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
#  Augmentation strategies
# ---------------------------------------------------------------------------

def _shuffle_options(ex: dict) -> dict:
    e = ex.copy()
    opts    = json.loads(e["options"])
    correct = set(json.loads(e["correct_options"]))
    random.shuffle(opts)
    e["options"]         = json.dumps(opts)
    e["correct_options"] = json.dumps([o for o in opts if o in correct])
    # Rebuild prompt with new option order
    e["prompt"] = _rebuild_prompt(e)
    e["augmentation"] = "option_shuffle"
    e["id"] = _make_id(e["prompt"])
    return e


def _rephrase_question(ex: dict) -> dict:
    e = ex.copy()
    q = e["question"]
    for orig, repls in PARAPHRASE_MAP.items():
        if orig in q:
            q = q.replace(orig, random.choice(repls), 1)
    e["question"] = q
    e["prompt"]   = _rebuild_prompt(e)
    e["augmentation"] = "question_rephrase"
    e["id"] = _make_id(e["prompt"])
    return e


def _change_difficulty(ex: dict, difficulty: str) -> dict:
    e = ex.copy()
    e["difficulty"]   = difficulty
    e["augmentation"] = f"difficulty_{difficulty}"
    e["id"] = _make_id(e["prompt"] + difficulty)
    return e


def _ttp_perspective(ex: dict, perspective: str) -> dict:
    e = ex.copy()
    prefix    = f"[Perspective: {perspective}]\n"
    e["prompt"]       = prefix + e["prompt"]
    e["augmentation"] = "perspective_" + perspective.replace(" ", "_")
    e["id"] = _make_id(e["prompt"])
    return e


def _negation_focus(ex: dict) -> dict:
    """Create a 'why are the other options wrong?' variant."""
    e = ex.copy()
    opts    = json.loads(e["options"])
    correct = set(json.loads(e["correct_options"]))
    wrong   = [o for o in opts if o not in correct]
    if not wrong:
        return e
    e["question"] = (
        "Which of the following options does the report NOT support? "
        + e["question"]
    )
    e["correct_options"] = json.dumps(wrong[:1])
    e["prompt"]          = _rebuild_prompt(e)
    e["augmentation"]    = "negation_focus"
    e["difficulty"]      = "hard"
    e["id"] = _make_id(e["prompt"])
    return e


def augment_example(seed: dict) -> list[dict]:
    """Return ~16 deterministic variants of a single seed."""
    variants: list[dict] = [seed]

    # Option shuffles x4
    for _ in range(4):
        variants.append(_shuffle_options(seed))

    # Question rephrase x3
    for _ in range(3):
        variants.append(_rephrase_question(seed))

    # Difficulty relabel x2
    for d in DIFFICULTY_LEVELS:
        if d != seed.get("difficulty"):
            variants.append(_change_difficulty(seed, d))

    # TTP-perspective x5
    for p in TTP_PERSPECTIVES:
        variants.append(_ttp_perspective(seed, p))

    # Negation focus x1
    variants.append(_negation_focus(seed))

    return variants


# ---------------------------------------------------------------------------
#  Augment to target
# ---------------------------------------------------------------------------

def augment_to_target(seeds_path: str, out_dir: str, target: int) -> str:
    seeds: list[dict] = []
    with open(seeds_path) as f:
        for line in f:
            seeds.append(json.loads(line))

    log.info("Loaded %d seeds. Augmenting to %d examples…", len(seeds), target)

    seen:  set[str] = set()
    rows:  list[dict] = []

    pbar = tqdm(total=target, desc="Augmenting")
    cycle = 0

    while len(rows) < target:
        cycle += 1
        random.shuffle(seeds)
        for seed in seeds:
            if len(rows) >= target:
                break
            for variant in augment_example(seed):
                if len(rows) >= target:
                    break
                uid = variant["id"]
                if uid in seen:
                    # Add a cycle salt to create a fresh unique ID
                    variant = variant.copy()
                    variant["id"] = _make_id(variant["id"] + str(cycle))
                    variant["augmentation"] = variant["augmentation"] + f"_c{cycle}"
                if variant["id"] in seen:
                    continue
                seen.add(variant["id"])
                rows.append(variant)
                pbar.update(1)

    pbar.close()

    out_path = Path(out_dir) / "dataset_1m.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")

    log.info("Wrote %d examples -> %s", len(rows), out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
#  HuggingFace push
# ---------------------------------------------------------------------------

def push_to_hub(
    dataset_path: str,
    repo_id: str,
    private: bool = False,
) -> None:
    log.info("Loading %s…", dataset_path)
    rows: list[dict] = []
    with open(dataset_path) as f:
        for line in tqdm(f, desc="Reading JSONL"):
            try:
                rows.append(_normalise_row(json.loads(line)))
            except Exception:
                continue

    log.info("Loaded %d rows. Splitting…", len(rows))
    random.shuffle(rows)
    n      = len(rows)
    n_val  = max(500, int(n * 0.01))
    n_test = max(500, int(n * 0.01))

    ds = DatasetDict({
        "train":      Dataset.from_list(rows[n_val + n_test:], features=HF_FEATURES),
        "validation": Dataset.from_list(rows[:n_val],          features=HF_FEATURES),
        "test":       Dataset.from_list(rows[n_val:n_val+n_test], features=HF_FEATURES),
    })

    log.info("Splits: %s", {k: len(v) for k, v in ds.items()})
    ds.push_to_hub(repo_id, private=private)
    log.info(
        "Published! https://huggingface.co/datasets/%s", repo_id
    )


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def _infer_difficulty(result: dict) -> str:
    n_correct = len(result.get("correct_options", []))
    n_ttps    = len(result.get("ttps", []))
    if n_correct > 1 or n_ttps > 3:
        return "hard"
    if n_ttps >= 2:
        return "medium"
    return "easy"


def _rebuild_prompt(ex: dict) -> str:
    """Re-format the prompt from stored fields."""
    opts = json.loads(ex["options"])
    opts_str = "\n".join(f"  {chr(65+i)}. {o}" for i, o in enumerate(opts))
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\nQUESTION: {ex['question']}\nOPTIONS:\n{opts_str}\n"
    )

def _build_row(result: dict, report: dict, source: str, difficulty: str, augmentation: str) -> dict:
    """Convert a Gemini distillation result into a HF-ready row."""
    q       = result.get("question", "")
    options = result.get("options", [])
    correct = result.get("correct_options", [])
    thought = result.get("thought", "<thought>\n" + result.get("rationale", "") + "\n</thought>")
    answer  = result.get("answer",  "<answer>" + ", ".join(correct) + "</answer>")

    user_msg = (
        f"REPORT:\n{json.dumps(report, indent=2)}\n\n"
        f"QUESTION: {q}\n"
        f"OPTIONS:\n" + "\n".join(f"  {chr(65+i)}. {o}" for i, o in enumerate(options))
    )
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_msg}\n"
    
    return {
        "id": _make_id(prompt),
        "prompt": prompt,
        "thought": thought,
        "answer": answer,
        "full_response": f"{thought}\n{answer}",
        "question": q,
        "options": json.dumps(options),
        "correct_options": json.dumps(correct),
        "malware_family": result.get("malware_family", "Unknown"),
        "ttps": json.dumps(result.get("ttps", [])),
        "difficulty": difficulty,
        "source": source,
        "augmentation": augmentation,
    }

def _normalise_row(row: dict) -> dict:
    """Ensure all fields are strings as required by HF_FEATURES."""
    res = {}
    for k in HF_FEATURES.keys():
        val = row.get(k, "")
        if isinstance(val, list):
            val = json.dumps(val)
        elif not isinstance(val, str):
            val = str(val)
        res[k] = val
    return res


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build 1M-sample malware RL dataset.")
    parser.add_argument("--mode", choices=["seed", "augment", "push", "all"], required=True)
    parser.add_argument("--reports", type=str, help="Dir of raw Hybrid-Analysis reports")
    parser.add_argument("--seeds", type=str, help="Path to seeds.jsonl")
    parser.add_argument("--dataset", type=str, help="Path to augmented JSONL dataset")
    parser.add_argument("--out", type=str, default="data/hf_dataset", help="Output dir")
    parser.add_argument("--limit", type=int, default=10_000, help="Max seed reports to process")
    parser.add_argument("--target", type=int, default=1_000_000, help="Target generated samples")
    parser.add_argument("--repo", type=str, help="HF HuggingFace repo ID (e.g. user/dataset)")
    parser.add_argument("--private", action="store_true", help="Make HF dataset private")

    args = parser.parse_args()

    if args.mode in ("seed", "all"):
        if not args.reports:
            parser.error("--reports required for seed generation")
        seeds_path = generate_seeds(args.reports, args.out, limit=args.limit)
    else:
        seeds_path = args.seeds

    if args.mode in ("augment", "all"):
        if not seeds_path:
            parser.error("--seeds required for augmentation (or use --mode all)")
        dataset_path = augment_to_target(seeds_path, args.out, args.target)
    else:
        dataset_path = args.dataset

    if args.mode in ("push", "all"):
        if not args.repo:
            parser.error("--repo required for HF push")
        if not dataset_path:
            parser.error("--dataset required for push")
        push_to_hub(dataset_path, args.repo, args.private)

if __name__ == "__main__":
    main()

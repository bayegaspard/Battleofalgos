import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os

def train():
    # Detect hardware
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    
    # Use smaller model for Mac to avoid OOM
    model_id = "mistralai/Mistral-7B-Instruct-v0.3" 
    if device == "mps":
        model_id = "unsloth/Llama-3.2-1B-Instruct" # Lightweight for local dev
        print(f"CUDA NOT DETECTED. Using lightweight model for Mac: {model_id}")

    base_dir = os.getcwd()
    dataset_path = os.path.join(base_dir, "data/finetuning/malware_sft_data.jsonl")
    output_dir = os.path.join(base_dir, "research/results/sft_mistral_lora")

    print(f"Dataset path: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"CRITICAL ERROR: Dataset not found at {dataset_path}")
        print(f"Current working directory: {base_dir}")
        print("Please ensure you have run the distillation step (Stage 2).")
        return

    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load in 4-bit/8-bit if using BitsAndBytes (CUDA only) or just FP16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device != "cuda":
        model = model.to(device)

    # LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def format_instruction(sample):
        # The dataset has 'instruction', 'input', 'output' keys
        return f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n{sample['output']}"

    def tokenize_function(examples):
        texts = [format_instruction({k: examples[k][i] for k in examples}) for i in range(len(examples['input']))]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=1024)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True if device == "cuda" else False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Starting training...")
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train()

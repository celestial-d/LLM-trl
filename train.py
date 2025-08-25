import os
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Callback for logging training progress
class LogCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            logger.info(f"Step {state.global_step}: loss = {state.loss}")

# Custom dataset
class CodeAlpacaDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.samples = []
        for i, item in enumerate(data):
            instruction = item["instruction"]
            input_text = item.get("input", "")
            output = item["output"]

            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Output:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Output:\n{output}"

            encoded = tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

            if i % 1000 == 0:
                logger.info(f"Processed {i} samples...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def format_prompt(example):
    if example.get("input", ""):
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"
    else:
        return f"### Instruction:\n{example['instruction']}\n\n### Output:\n{example['output']}"


# Load tokenizer and model
model_name = "facebook/opt-1.3b"
logger.info(f"Loading model and tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
logger.info("Loading dataset...")
raw_dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
raw_dataset = raw_dataset.map(lambda x: {"text": format_prompt(x)})
logger.info("Loaded dataset successfully")
tokenized_dataset = raw_dataset
# Use subset for quick run
subset = raw_dataset.select(range(5000))
train_dataset = CodeAlpacaDataset(subset, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./opt125m-codealpaca-sft",
    per_device_train_batch_size=400,
    gradient_accumulation_steps=400,
    num_train_epochs=1,
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),
    save_strategy="no",
    logging_steps=10,
)

# Initialize trainer
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized_dataset = raw_dataset.map(tokenize, batched=True)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=TrainingArguments(
        output_dir="./opt125m-codealpaca-sft",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        learning_rate=2e-5,
        report_to=[],  # disables wandb
    )
)

logger.info("Starting training...")
trainer.train()
logger.info("Training completed.")

# Set model to evaluation mode
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Create a DataLoader from the training dataset (or use a held-out portion)
def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }

eval_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn)

total_tokens = 0
correct_tokens = 0

with torch.no_grad():
    for batch in tqdm(eval_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(model.device)
        labels = batch["labels"].to(model.device)

        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Take argmax over vocab dimension
        predictions = torch.argmax(logits, dim=-1)

        # Only count non-padding positions
        mask = labels != -100  # make sure your trainer uses label=-100 for ignored positions
        correct = (predictions == labels) & mask

        correct_tokens += correct.sum().item()
        total_tokens += mask.sum().item()

accuracy = correct_tokens / total_tokens
print(f"\n Token-level Accuracy: {accuracy * 100:.2f}%")

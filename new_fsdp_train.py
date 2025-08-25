#!/usr/bin/env python
# new_fsdp_train.py
import json, math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

def format_row(ex):
    instr = (ex.get("instruction") or "").strip()
    inp   = (ex.get("input") or "").strip()
    out   = (ex.get("output") or "").strip()
    if inp:
        return f"### Question:\n{instr}\n\n### Input:\n{inp}\n\n### Answer:\n{out}"
    return f"### Question:\n{instr}\n\n### Answer:\n{out}"

def main():
    # -------- sane defaults (2 GPUs, no LoRA/quant) --------
    model_name = "facebook/opt-125m"
    dataset_name = "sahil2801/CodeAlpaca-20k"
    output_dir = "./opt27b-codealpaca-fsdp"
    max_seq_length = 1024          # safer default on 2×24–40GB; raise to 2048 if you have headroom
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 16
    num_train_epochs = 3
    learning_rate = 2e-5
    weight_decay = 0.0
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"
    logging_steps = 10
    save_steps = 500
    save_total_limit = 3
    eval_steps = 250               # periodic eval on train set
    seed = 42

    # precision & memory
    bf16 = True                    # switch to fp16=True and bf16=False if your GPUs lack bfloat16
    fp16 = False
    gradient_checkpointing = True
    attn_impl = "sdpa"             # use "eager" if SDPA is unstable on your stack
    packing = True

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # -------- dataset (train == eval) --------
    raw = load_dataset(dataset_name, split="train")
    raw = raw.shuffle(seed=seed)
    train = raw.map(lambda ex: {"text": format_row(ex)}, remove_columns=raw.column_names)
    eval_ds = train  # evaluation dataset is the same as training dataset

    # -------- tokenizer --------
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # -------- model --------
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(torch.bfloat16 if bf16 else torch.float16 if fp16 else None),
        attn_implementation=attn_impl,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optim_name = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

    # -------- Trainer args (with periodic eval) --------
    targs = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        eval_accumulation_steps=1,
        bf16=bf16,
        fp16=fp16,
        optim=optim_name,
        report_to=["none"],
        seed=seed,
        ddp_find_unused_parameters=False,
    )

    # -------- SFTTrainer --------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train,
        eval_dataset=eval_ds,             # same as train
        args=targs,
        max_seq_length=max_seq_length,
        packing=packing,
        dataset_text_field="text",
    )

    # train + periodic eval (on train set)
    trainer.train()

    # final eval via SFTTrainer.evaluate()
    metrics = trainer.evaluate()
    if "eval_loss" in metrics and metrics["eval_loss"] is not None:
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
    with open(Path(output_dir) / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Final eval (train-as-eval):", metrics)

    # save final artifacts
    # trainer.save_model(output_dir)
    # tok.save_pretrained(output_dir)

if __name__ == "__main__":
    main()

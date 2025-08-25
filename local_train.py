#!/usr/bin/env python
# local_train.py — single-GPU dense SFT debug run (no LoRA, no quant)
# Mirrors your Flower client logic to reproduce non-zero loss locally.

import os
import time
import argparse
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import TrainingArguments, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# Reuse your project code
from llm.models import get_model
from llm.dataset import formatting_prompts_func  # must match your client formatter


# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_dict_of_lists(batch_list):
    """Convert List[Dict[str, Any]] -> Dict[str, List[Any]]"""
    if not batch_list:
        return {}
    keys = batch_list[0].keys()
    return {k: [row[k] for row in batch_list] for k in keys}


def build_tokenizer_and_collator(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Must EXACTLY match your formatter’s template
    response_template_with_context = "\n### Response:"
    response_template_ids = tok.encode(
        response_template_with_context, add_special_tokens=False
    )  # <-- no slicing

    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tok)
    return tok, collator


def sanity_check_labels(tokenizer, data_collator, fmt_fn, dataset, seq_len=512, n=4):
    """
    Ensure some labels are not -100 (template matched and supervision present).
    Raises if all labels are masked.
    """
    from torch.utils.data import DataLoader

    take = min(n, len(dataset))
    if take == 0:
        raise RuntimeError("Sanity check failed: dataset is empty")

    subset = dataset.select(range(take))

    def _cf(batch_list):
        # batch_list is List[Dict[str, Any]] from DataLoader
        batch_dict = _as_dict_of_lists(batch_list)         # formatter expects dict-of-lists
        texts = fmt_fn(batch_dict)                         # -> List[str]
        enc = tokenizer(                                   # tokenize the list of texts
            texts,
            padding=True,
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        # Split BatchEncoding into a list[dict[str, tensor]] for the collator
        examples = [{k: enc[k][i] for k in enc} for i in range(enc["input_ids"].size(0))]
        return data_collator(examples)

    dl = DataLoader(subset, batch_size=min(2, take), shuffle=False, collate_fn=_cf)
    batch = next(iter(dl))
    labels = batch.get("labels", None)
    if labels is None:
        raise RuntimeError("Sanity check failed: 'labels' missing from collated batch")
    supervised = (labels != -100).sum().item()
    print(f"[sanity] supervised tokens in first batch: {supervised}")
    if supervised == 0:
        raise RuntimeError(
            "All labels are -100 (masked). Check the response template and formatter."
        )


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
    parser.add_argument("--dataset_name", type=str, default="vicgalle/alpaca-gpt4")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--limit_train_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load dataset (single local partition). For Alpaca-GPT4 the column is "output"; rename to "response".
    ds = load_dataset(args.dataset_name, split="train")
    if "response" not in ds.column_names and "output" in ds.column_names:
        ds = ds.rename_column("output", "response")
    if args.limit_train_samples and len(ds) > args.limit_train_samples:
        ds = ds.select(range(args.limit_train_samples))

    tokenizer, data_collator = build_tokenizer_and_collator(args.model_name)

    # Build dense model via your project loader (no LoRA/quant)
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.create({
        "name": args.model_name,
        "dtype": args.dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "attn_implementation": args.attn_impl,
    })
    model = get_model(model_cfg)

    # Output dir: unique per run; disable resume to avoid 0.0 losses from auto-resume
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.output_dir or os.path.join("out_local", f"{os.path.basename(args.model_name)}_{ts}")
    os.makedirs(outdir, exist_ok=True)

    # TrainingArguments (single GPU). Align AMP flags with dtype.
    fp16 = args.dtype == "fp16"
    bf16 = args.dtype == "bf16"
    optim = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=outdir,
        overwrite_output_dir=True,         # ensure fresh run
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,          # takes precedence over epochs if > 0
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim=optim,
        report_to=[],                      # no external loggers by default
    )

    # Sanity-check masking before training
    sanity_check_labels(
        tokenizer, data_collator, formatting_prompts_func, ds, seq_len=args.seq_length
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=args.seq_length,
        train_dataset=ds,
        formatting_func=formatting_prompts_func,  # your formatter
        data_collator=data_collator,
    )

    print(
        f"[info] starting train: outdir={outdir}, max_steps={args.max_steps}, "
        f"bs={args.per_device_train_batch_size}x{args.gradient_accumulation_steps}, dtype={args.dtype}"
    )
    out = trainer.train(resume_from_checkpoint=False)  # force fresh steps

    # Robust loss extraction
    training_loss = (
        float(out.training_loss)
        if getattr(out, "training_loss", None) is not None
        else float(out.metrics.get("train_loss", 0.0))
    )
    print(f"[done] training_loss={training_loss:.6f}")
    print(f"[metrics] {out.metrics}")

    # Save model for inspection
    trainer.save_model(outdir)
    print(f"[save] model saved to: {outdir}")


if __name__ == "__main__":
    main()

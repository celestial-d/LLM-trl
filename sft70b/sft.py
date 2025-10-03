#!/usr/bin/env python3
import os
import socket
import torch
import shutil
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTConfig, SFTTrainer
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity

# ---------- GCC Utils ----------
def _which_many(cands):
    for c in cands:
        p = shutil.which(c)
        if p:
            return p
        if os.path.isabs(c) and os.path.exists(c):
            return c
    return None

gcc_candidates = [
    "g++-12", "gcc-12", "/usr/bin/g++-12", "/usr/local/bin/g++-12",
    "/soft/compilers/gcc/12.3.0/bin/g++", "/soft/compilers/gcc/12.3.0/bin/gcc",
    "/opt/gcc/12.3.0/bin/g++", "/opt/gcc/12.3.0/bin/gcc",
]
gxx = _which_many(gcc_candidates)
gcc = gxx  # ok to set both to the same path; torch checks CXX
if gxx:
    os.environ["CXX"] = gxx
    os.environ["CC"]  = gcc
    print(f"[ds] Using CXX={gxx}")
else:
    # Fall back to whatever is present; we’ll also prepare a DS config fallback below
    print("WARNING: Could not find GCC/G++ 12; will enable DS fallback if needed")

# ---------- Utils ----------
def get_global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    for k in ("RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK"):
        v = os.getenv(k)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0

def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    for k in ("WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE"):
        v = os.getenv(k)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 1

# ---------- Profiler helpers ----------
class StepTraceHandler:
    """
    After each active window, write both Chrome trace and TensorBoard events to:
      ./llama3_wikitext_profiler/step_{N:06d}/rank{R:03d}/
    """
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.emit_idx = 0

    def __call__(self, prof):
        rank = get_global_rank()
        step_dir = os.path.join(self.base_dir, f"step_{self.emit_idx:06d}", f"rank{rank:03d}")
        os.makedirs(step_dir, exist_ok=True)

        # Chrome trace
        prof.export_chrome_trace(os.path.join(step_dir, "trace.json"))


        # Optional: tiny metadata file
        try:
            with open(os.path.join(step_dir, "meta.txt"), "w") as f:
                f.write(
                    f"host={socket.gethostname()}\n"
                    f"pid={os.getpid()}\n"
                    f"rank={rank}\n"
                    f"world_size={get_world_size()}\n"
                )
        except Exception:
            pass

        self.emit_idx += 1

class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof
    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

def main():
    # ---------- TRL / SFT config ----------
    training_args = SFTConfig(
        output_dir="llama3-wikitext-SFT_2",
        bf16=True,
        use_liger_kernel=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=512,
        packing=True,
        per_device_train_batch_size=16,   #16 for 8B, 2 for 70B
        gradient_accumulation_steps=16,    #16 for 8B, 8 for 70B
        dataset_num_proc=64,          #64 for 8B, 32 for 70B
        num_train_epochs=1,
    )

    # ---------- Model & tokenizer ----------
    # model_path = os.path.expanduser("../../models/Qwen3-8B")
    #model_path = os.path.join(os.environ["SCRATCH"], "Llama-3.1-70B")
    #model_path = "/lus/eagle/projects/SR-APPFL/duo/models/llama-3p3-70b-instruct"
    scratch = os.getenv("SCRATCH", "/lus/eagle/projects/SR-APPFL/duo")
    model_path = os.path.join(scratch, "models/llama3-8b-instruct")
    #model_path = os.path.join(scratch, "models/llama-3p3-70b-instruct")
    # or the absolute path directly if that’s where it lives

    #model_path = "/lus/eagle/projects/SR-APPFL/duo/models/llama3-8b-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # ---------- Dataset ----------
    data_dir = "/lus/eagle/projects/SR-APPFL/duo/LLM-trl/sft70b/wikitext"
    raw_ds = load_dataset(
        "parquet",
        data_files={
            "train": [
                f"{data_dir}/wikitext-103-v1/train-00000-of-00002.parquet",
            ]
        },
        split="train",
    )

    # ---------- Trainer ----------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=raw_ds,
    )

    # # ---------- Profiler setup (CURRENT DIR) ----------
    # prof_base = os.path.join(os.getcwd(), "llama3_wikitext_profiler")
    # step_handler = StepTraceHandler(prof_base)

    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()

    # # One-step windows; repeat effectively forever (until training ends)
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule = schedule(skip_first=0, wait=50, warmup=0, active=20, repeat=8),
    #     on_trace_ready=step_handler,
    #     profile_memory=False,
    #     with_stack=False,
    #     record_shapes=False,
    # ) as prof:
    #     trainer.add_callback(ProfCallback(prof=prof))
    #     trainer.train()
    results = trainer.train()
    if get_global_rank() == 0:
        print(f"[rank0] Train done. train_loss={results.training_loss}.")


if __name__ == "__main__":
    main()


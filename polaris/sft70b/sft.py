#!/usr/bin/env python3
import os
import socket
import torch
import shutil
import torch.distributed as dist
#from datasets import load_dataset
from dataset import load_data
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTConfig, SFTTrainer
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity

def get_local_rank():
    # Common envs from torchrun/DeepSpeed/SLURM/OpenMPI/PMI
    for k in ("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", "PMI_LOCAL_RANK", "MPI_LOCALRANKID"):
        v = os.environ.get(k)
        if v is not None:
            return int(v)
    # Derive from global rank if needed
    try:
        r = int(os.environ.get("RANK", "0"))
        nppn = int(os.environ.get("NPROC_PER_NODE", os.environ.get("LOCAL_SIZE", "1")))
        return r % max(nppn, 1)
    except Exception:
        pass
    # Fall back to CUDA device index
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.current_device()
    except Exception:
        pass
    return 0


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


def formatting_prompts_func(example):
    header = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    # If TRL ever passes a batch (rare), handle it gracefully:
    if isinstance(example.get("instruction"), list):
        out = []
        N = len(example["instruction"])
        inputs = example.get("input", [None] * N)
        for i in range(N):
            instr = (example["instruction"][i] or "").strip()
            resp  = (example["response"][i] or "").strip()
            inp   = (inputs[i] or "").strip() if inputs is not None else ""
            if inp:
                s = f"{header}\n### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response: {resp}"
            else:
                s = f"{header}\n### Instruction:\n{instr}\n\n### Response: {resp}"
            out.append(s)
        return out  # OK only if TRL actually batched

    # Normal path: single example -> return ONE string
    instr = (example.get("instruction") or "").strip()
    resp  = (example.get("response") or "").strip()
    inp   = (example.get("input") or "").strip() if example.get("input") is not None else ""
    if inp:
        return f"{header}\n### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response: {resp}"
    else:
        return f"{header}\n### Instruction:\n{instr}\n\n### Response: {resp}"


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof
    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

def main():
    # ---------- TRL / SFT config ----------
    training_args = SFTConfig(
        output_dir="/lus/eagle/projects/SR-APPFL/duo/tmp/llama3-wikitext-SFT_2",
        bf16=True,
        use_liger_kernel=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=512,
        packing=True,
        per_device_train_batch_size=2,   #16 for 8B, 2 for 70B
        gradient_accumulation_steps=8,    #16 for 8B, 8 for 70B
        dataset_num_proc=32,          #64 for 8B, 32 for 70B      
        save_strategy="no",
    ) #num_train_epochs=1,

    # ---------- Model & tokenizer ----------
    # model_path = os.path.expanduser("../../models/Qwen3-8B")
    #model_path = os.path.join(os.environ["SCRATCH"], "Llama-3.1-70B")
    #model_path = "/lus/eagle/projects/SR-APPFL/duo/models/llama-3p3-70b-instruct"
    scratch = os.getenv("SCRATCH", "/lus/eagle/projects/SR-APPFL/duo")
    model_path = os.path.join(scratch, "models/llama3-8b-instruct")
    #model_path = os.path.join(scratch, "llama-3p3-70b-instruct")
    # or the absolute path directly if that’s where it lives

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # ---------- Dataset ----------
    # data_dir = "/lus/eagle/projects/SR-APPFL/duo/LLM-trl/sft70b/wikitext"
    # raw_ds = load_dataset(
    #     "parquet",
    #     data_files={
    #         "train": [
    #             f"{data_dir}/wikitext-103-v1/train-00000-of-00002.parquet",
    #         ]
    #     },
    #     split="train",
    # )
    
    #trainset = load_data(partition_id, num_partitions, dataset_name)
    trainset = load_data(0, 2, "/home/zhangduo4610/CodeAlpaca-20k")
    # ---------- Trainer ----------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=trainset,
        formatting_func=formatting_prompts_func,
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
    if get_local_rank() == 0:
        print(f"[rank0] Train done. train_loss={results.training_loss}.")


if __name__ == "__main__":
    main()


#!/bin/bash
# ===== Polaris PBS job: N nodes, 4 GPUs per node =====
#PBS -A SR-APPFL
#PBS -q debug
#PBS -l select=2:system=polaris            
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -N 8cds_ds_8
#PBS -j oe
#PBS -V

set -euxo pipefail
# Make g++ less strict about the enum types Triton’s helper uses (helps on some toolchains)
export CXXFLAGS="${CXXFLAGS:-} -fpermissive"

# # --- Paths / env ---
ENV_NAME_PATH="/lus/eagle/projects/SR-APPFL/duo/env/sft"
APP_DIR="/lus/eagle/projects/SR-APPFL/duo/LLM-trl/polaris"
SCRIPT_PATH="${APP_DIR}/sft.py"
DS_CFG="${APP_DIR}/deepspeed_zero3.yaml"

# --- Triton / DS knobs for A100s and toolchain ---
export TRITON_DISABLE_TMA=1                         # A100: avoid TMA (CUtensorMap) build path
export CXXFLAGS="${CXXFLAGS:-} -fpermissive"        # safe with `set -u`
export DS_BUILD_SPARSE_ATTN=0
export DS_SKIP_CUDA_BUILD=1


# # --- Polaris modules ---
# module use /soft/modulefiles
# module load gcc-native/12.3
# module load conda/2024-04-29-aws-nccl
# module load cudatoolkit-standalone/12.4.0
# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
# conda activate "$ENV_NAME_PATH"

# --- Node / world sizing ---
export GPUS_PER_NODE=4
export NNODES=$(wc -l < "$PBS_NODEFILE")
export WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))

# Head node for rendezvous
HEADNODE=$(head -n1 "$PBS_NODEFILE")
# If you prefer an IP:
# HEADIP=$(getent hosts "$HEADNODE" | awk '{print $1}')
HEADIP="$HEADNODE"
export RDZV_PORT=12355

# --- Runtime hygiene / caches (per-rank) ---
export OMP_NUM_THREADS=8
export TMPDIR="/tmp/${USER}_${PBS_JOBID%%.*}"
mkdir -p "$TMPDIR"

# Use PMI_RANK for uniqueness under mpiexec (PBS/MPICH)
export TRITON_CACHE_DIR_BASE="$TMPDIR/triton"
export TORCH_EXTENSIONS_DIR_BASE="$TMPDIR/torch_ext"
mkdir -p "$TRITON_CACHE_DIR_BASE" "$TORCH_EXTENSIONS_DIR_BASE"

# add near the top, after ENV_NAME_PATH
PY="${ENV_NAME_PATH}/bin/python"

# Optional Polaris comms (uncomment if you’ve been using Libfabric/CXI)
# export FI_PROVIDER=cxi
# export FI_CXI_DISABLE_HOST_REGISTER=1
# export FI_MR_CACHE_MONITOR=userfaultfd
# export NCCL_NET="AWS Libfabric"
# export NCCL_IB_DISABLE=1
# export NCCL_CROSS_NIC=1
# export NCCL_COLLNET_ENABLE=0
# export NCCL_SHM_DISABLE=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET

export PYTHONNOUSERSITE=1


# ---- Build accelerate launch line (runs once per node) ----
LAUNCHER='
  export RANK=${PMI_RANK:-0} ;
  export LOCAL_RANK=${PMI_LOCAL_RANK:-$((RANK % '"$GPUS_PER_NODE"'))} ;
  export TRITON_CACHE_DIR='"$TRITON_CACHE_DIR_BASE"'/$RANK ;
  export TORCH_EXTENSIONS_DIR='"$TORCH_EXTENSIONS_DIR_BASE"'/$RANK ;
  mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" ;

# --- Polaris modules ---
  module use /soft/modulefiles
  module load gcc-native/12.3
  module load conda/2024-04-29-aws-nccl
  module load cudatoolkit-standalone/12.4.0
  unset http_proxy https_proxy HTTP_PROXY
  conda activate /lus/eagle/projects/SR-APPFL/duo/env/sft

  HF_HUB_ENABLE_HF_TRANSFER=1 \
  ACCELERATE_LOG_LEVEL=info \
  TRANSFORMERS_VERBOSITY=info \
  python -m accelerate.commands.launch \
    --config_file '"$DS_CFG"' \
    --num_machines '"$NNODES"' \
    --num_processes '"$WORLD_SIZE"' \
    --main_process_ip '"$HEADIP"' \
    --main_process_port '"$RDZV_PORT"' \
    --machine_rank ${PMI_RANK:-0} \
    --tee 3 \
    '"$SCRIPT_PATH"'
'


# ---- Kick off: one launcher per node ----
# -ppn 1 ensures exactly one accelerate launcher per node
mpiexec -n "$NNODES" -ppn 1 bash -lc "$LAUNCHER"





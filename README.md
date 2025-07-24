# LLM Training with TRL

CUDA_VISIBLE_DEVICES=0 python train.py

CUDA_VISIBLE_DEVICES=0,1 accelerate launch fsdp_train.py

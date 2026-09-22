#!/bin/bash

# Training script for NLST lung cancer prediction with DETR and Atlas backbone
# Multi-GPU training using uv and torchrun
#
# Usage:
#   bash scripts/run.sh                    # Run with default config
#   bash scripts/run.sh custom_config.yaml # Run with custom config

# Configuration
CONFIG_FILE="${1:-configs/nlst_detr_atlas.yaml}"
NOTES=""
EXP_NAME="NLST_FT_Atlas_unimodal_all_lr1.e-5_warmup4_wd0_stages228_ps884_ep11_detr_sybil_cosine50-exp005"
TAGS='["apply-pooling", "DETR3D", "rve"]'

# Training settings
NUM_GPUS=8
MASTER_PORT=2300
OMP_NUM_THREADS=2

# Multi-GPU training with uv and torchrun
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OMP_NUM_THREADS=$OMP_NUM_THREADS \
uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    $CONFIG_FILE \
    --opts \
    experiment.name "$EXP_NAME" \
    experiment.tags "$TAGS" \
    experiment.notes "$NOTES" \
    engine.max_epochs 50

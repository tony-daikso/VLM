#!/bin/bash
# Full run: 950k CG reports, abdomen -> chest (brain skipped for now; add "brain" to --types to run it later). Resumable: just run again.
source /root/miniconda3/etc/profile.d/conda.sh && conda activate vllm
cd /datadrive/VLM/data/CT/CG/radar_preprocess/scripts
curl -s localhost:11434/api/generate -d '{"model":"qwen3.8:27b","keep_alive":0}' >/dev/null  # free GPU from ollama
VLLM_USE_FLASHINFER_SAMPLER=0 python radar_llm_preprocess_combined.py \
    --types abdomen chest --chunk 2000 >> ../logs/full.log 2>&1
echo "EXIT $?" >> ../logs/full.log

#!/usr/bin/env bash

set -euo pipefail


# Modify the following environment variables as needed
export HF_HOME="<HF_CACHE_DIR>"  # e.g., /data/huggingface_cache
export CUDA_VISIBLE_DEVICES="<CUDA_DEVICES>"  # e.g., 0 or 0,1
# If you need to specify CUDA, please export CUDA_HOME=/usr/local/cuda-12.4 and update PATH externally
# export PATH="$CUDA_HOME/bin:$PATH"

# Parameters configuration (modify as needed)
MODALITY="MRI"   # Optional: CT, Ultrasound, MRI, Xray, Dermoscopy, Microscopy, Fundus
PROMPT_TYPE="complex"  # Optional: simple, complex
MODEL_PATH="<MODEL_REPO_OR_DIR>"  # e.g., JZPeterPan/MedVLM-R1 or /path/to/checkpoint
BSZ=1
PROMPT_PATH="MRI_CT_XRAY_300each_dataset.json"
BASE_PATH="<DATASET_PATH_ROOT>/Huatuo"
MAX_NEW_TOKENS=512
DO_SAMPLE=false
TEMPERATURE=1.0

# Output path
OUTPUT_DIR="<OUTPUT_DIR>"  # e.g., /data/outputs/logs/medvlm-r1
mkdir -p "$OUTPUT_DIR"
OUTPUT_PATH="$OUTPUT_DIR/Ours_MRI1600_OOD_${MODALITY}300.json"

# Run Python script
python3 src/eval/test_qwen2vl_med.py \
  --modality "$MODALITY" \
  --prompt_type "$PROMPT_TYPE" \
  --model_path "$MODEL_PATH" \
  --bsz "$BSZ" \
  --output_path "$OUTPUT_PATH" \
  --prompt_path "$PROMPT_PATH" \
  --base_path "$BASE_PATH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --do_sample "$DO_SAMPLE" \
  --temperature "$TEMPERATURE"



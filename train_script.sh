cd src/open-r1-multimodal

export DATASET="<DATASET_NAME>" # e.g., Huatuo, ISIC, SLAKE, VQA-RAD
export DATASET_SELECTION="MR" # Options: MR, CT, XRAY, COMBINED
export GPU_NUM="<GPU_NUM>" # e.g., 2
export DEBUG_MODE="<true_or_false>" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="<LOG_PATH>" # e.g., /data/outputs/${DATASET}_log_${GPU_NUM}GPU.txt
# export CUDA_HOME=/usr/local/cuda-12.2
# export PATH=$CUDA_HOME/bin:$PATH
export HF_HOME="<HF_CACHE_DIR>" # e.g., /data/huggingface_cache
export WANDB_ENTITY="<WANDB_ENTITY>"
export WANDB_PROJECT="<WANDB_PROJECT>"
export WANDB_RUN_GROUP="$DATASET"
export WANDB_JOB_TYPE="<WANDB_JOB_TYPE>" # e.g., training
export WANDB_RUN_NAME="<WANDB_RUN_NAME_PREFIX>-$(date +%Y-%m-%d-%H-%M-%S)"
# export CUDA_VISIBLE_DEVICES=5

torchrun --nproc_per_node="${GPU_NUM}" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="<MASTER_ADDR>" \
    --master_port="<MASTER_PORT>" \
    src/open_r1/grpo.py \
    --output_dir <OUTPUT_DIR_ROOT>/medvlm-r1/$WANDB_RUN_NAME \
    --model_name_or_path <MODEL_REPO_OR_DIR> \
    --dataset_name <DATASET_PATH_ROOT>/$DATASET/ \
    --dataset_selection $DATASET_SELECTION \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name $WANDB_RUN_NAME \
    --save_steps 50 \
    --save_only_model true \
    --num_generations 4

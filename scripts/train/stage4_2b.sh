#!/bin/bash
# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16667
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=4
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
echo $GRADIENT_ACCUMULATION_STEPS

# Log Arguments
export WANDB_PROJECT=videollama3_qwen2.5_2b
PRECEDING_RUN_NAME=stage_3
RUN_NAME=stage_4
DATA_DIR=/workspace/data_root
OUTP_DIR=/workspace

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    videollama3/train.py \
    --deepspeed scripts/zero1.json \
    --model_type videollama3_qwen2 \
    --model_path /workspace/videollama3_2b_local \
    --vision_encoder /workspace/VL3-SigLIP-NaViT \
    --mm_projector_type mlp2x_gelu \
    --data_path ${DATA_DIR}/stage3/stage3_selectTraindata.jsonl \
    --data_folder ${DATA_DIR} \
    --image_merge_size 2 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 180 \
    --model_max_length 16384 \
    --mm_max_length 10240 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --num_train_epochs 5 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --llm_lr 1e-5 \
    --mm_projector_lr 1e-5 \
    --vision_encoder_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --run_name $RUN_NAME \
    --dataset_cache_dir /mnt/damovl/DAMOVL_DATASETS/.cache
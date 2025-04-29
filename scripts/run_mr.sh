#!/bin/bash

version=v1
savepath="./save/$version"

dataset='./data/mrscore_dataset.json'
llm_model='ministral/Ministral-3b-instruct'

python -u train.py \
    --dataset $dataset \
    --llm_model $llm_model \
    --lora_inference False \
    --llm_r 32 \
    --llm_alpha 64 \
    --batch_size 8 \
    --val_batch_size 24 \
    --max_length 350 \
    --num_workers 8 \
    --learning_rate 0.00001 \
    --devices 1 \
    --accelerator gpu \
    --precision bf16-mixed \
    --num_nodes 1 \
    --strategy ddp \
    --max_epochs 30 \
    --accumulate_grad_batches 2 \
    --num_sanity_val_steps 0 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --savedmodel_path ${savepath} \
    2>&1 |tee -a ${savepath}/log.txt


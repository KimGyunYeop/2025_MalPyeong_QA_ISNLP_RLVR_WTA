#/bin/bash

wandb offline
export CUDA_VISIBLE_DEVICES=1

model_names=(
    "K-intelligence/Midm-2.0-Base-Instruct"
)

for model_name in "${model_names[@]}"; do
    python -u main.py \
        --model_name "$model_name" \
        --train \
        --num_train_epochs 15 \
        --lora_mode "lora" \
        --lora_r 16 \
        --lora_alpha 32 \
        --testing_every_epoch 5 \
        --general_prompt_path "prompts/공용프롬프트.txt" \
        --test \
        --run_name "finetuning/lora_15epoch" \
        --device "cuda:0"
done
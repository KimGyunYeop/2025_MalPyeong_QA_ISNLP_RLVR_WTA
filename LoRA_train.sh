#/bin/bash

wandb offline
export CUDA_VISIBLE_DEVICES=0

train_data_path="data/QA/korean_culture_qa_V1.0_train+.json" # change to your train data path
dev_data_path="data/QA/korean_culture_qa_V1.0_dev+.json" # change to your dev data path
test_data_path="data/QA/korean_culture_qa_V1.0_test+.json" # change to your test data path

model_names=(
    "K-intelligence/Midm-2.0-Base-Instruct"
)

for model_name in "${model_names[@]}"; do
    python -u main.py \
        --model_name "$model_name" \
        --train_data_path "$train_data_path" \
        --dev_data_path "$dev_data_path" \
        --test_data_path "$test_data_path" \
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
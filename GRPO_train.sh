#/bin/bash

wandb offline
export CUDA_VISIBLE_DEVICES=1

train_data_path="data/QA/korean_culture_qa_V1.0_train+.json" # change to your train data path
dev_data_path="data/QA/korean_culture_qa_V1.0_dev+.json" # change to your dev data path
test_data_path="data/QA/korean_culture_qa_V1.0_test+.json" # change to your test data path

model_names=(
    "K-intelligence/Midm-2.0-Base-Instruct"
    # "kakaocorp/kanana-1.5-8b-instruct-2505"
    # "skt/A.X-4.0-Light"
    # "trillionlabs/Tri-21B"
    # "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    # "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
)

for model_name in "${model_names[@]}"; do
    python -u main.py \
        --model_name "$model_name" \
        --train_data_path "$train_data_path" \
        --dev_data_path "$dev_data_path" \
        --test_data_path "$test_data_path" \
        --train \
        --lora_mode "lora" \
        --lora_r 16 \
        --lora_alpha 32 \
        --rl_mode "ppo" \
        --cand_temperature 1.0 \
        --cand_top_p 0.97 \
        --use_format_reward \
        --use_write_type_answer \
        --wta_reward_stretegy "cand_max" \
        --generation_reward_scale 5.0 \
        --reward_scale 1.0 \
        --test \
        --testing_every_epoch 1 \
        --run_name "analysis" \
        --device "cuda:0"
done
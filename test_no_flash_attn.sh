# !/bin/bash

export CUDA_VISIBLE_DEVICES=0

wandb offline
adapter_path="GyunYeop/midm-base-GRPO-tuning-KoreanCultureQA"

dev_data_path="data/QA/korean_culture_qa_V1.0_dev+.json" # change to your dev data path
test_data_path="data/QA/korean_culture_qa_V1.0_test+.json" # change to your test data path

python -u test.py \
    --adapter_path "$adapter_path" \
    --dev_data_path "$dev_data_path" \
    --test_data_path "$test_data_path" \
    --no_flash_attention \
    --num_beams 5 \
    --device "cuda:0"

# !/bin/bash

export CUDA_VISIBLE_DEVICES=0

wandb offline
adapter_path="GyunYeop/midm-base-GRPO-tuning-KoreanCultureQA"

test_data_path="data/QA/korean_culture_qa_V1.0_test+.json" # change to your test data path

python -u reproduce_test.py \
    --adapter_path "$adapter_path" \
    --test_data_path "$test_data_path" \
    --device "cuda:0"

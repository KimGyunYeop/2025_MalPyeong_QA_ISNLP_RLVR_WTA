#/bin/bash
wandb offline
export CUDA_VISIBLE_DEVICES=0

model_names=(
    "K-intelligence/Midm-2.0-Base-Instruct"
    "kakaocorp/kanana-1.5-8b-instruct-2505"
    "skt/A.X-4.0-Light"
    "trillionlabs/Tri-21B"
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
)

for model_name in "${model_names[@]}"; do
    python -u find_model.py \
        --model_name "$model_name" \
        --general_prompt_path "prompts/공용프롬프트.txt" \
        --run_name "findmodel" \
        --device "cuda:0"

    python -u find_model.py \
        --model_name "$model_name" \
        --general_prompt_path "prompts/COT공용프롬프트_for_find_model.txt" \
        --run_name "findmodel" \
        --device "cuda:0"
done
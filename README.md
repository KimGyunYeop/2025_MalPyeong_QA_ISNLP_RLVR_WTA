# 2025_MalPyeong_QA_ISNLP_RLVR_WTA
국립국어원 주최 2025년 인공지능의 한국어 능력 평가 경진대회 [2025]한국문화 질의응답(가 유형) ISNLP팀 공식 레포지토리

# Enviorment Setting

```
git clone https://github.com/KimGyunYeop/2025_MalPyeong_QA_ISNLP_RLVR_WTA.git
cd 2025_MalPyeong_QA_ISNLP_RLVR_WTA

conda create -n test python==3.12
conda activate test

pip install -r requirements.txt
#flash attention(flash-attn) issue 많음
pip install flash-attn==2.8.1 --no-build-isolation 
```
### !! flash attention

- flash-attn(pip install flash-attn --no-build-isolation)의 경우에는 사용자의 pc환경에 따라 다운로드 및 빌드가 매우 오래걸릴수 있다(본 참가팀은 4시간 소모되었으며 몇일 단위로 걸린다는 github issue 존재)
- flash-attn이 설치 불가능한 경우 --no_flash_attention을 argument로 입력하면 작동하지만 실험 시간이 증가하며 결과값이 미세하게 다르게나올 수 있다. (reproduce_test code 재현결과 test set의 서술형 200문항 중 25개의 답안에서 약간의 차이 발생)
- cuda가 root계정 위치에 설치되어있지 않으면 설치 불가
- flash attention의 경우 하드웨어 환경 및 가상환경에 따라 작동 방식 및 여부가 다른데, 이를 fallback으로 처리하는 과정에서 beam search 환경에서는 결과가 크게 달라지기도한다.
- **그렇기에 flash-attn 환경을 조성하기 힘들다면 점수 재현을 위한 실험에서는 일부 서술형 답안의 차이를 감안하더라도 flash attention을 사용하지 않는 환경을 추천한다.**

# Dataset Setting

본 실험은 2025년 인공지능의 한국어 능력 평가 경진대회 [2025]한국문화 질의응답(가 유형)에서 제공되는 데이터를 이용한다.

학습용은 주어진 train dataset만을 사용하였으며 dev dataset으로 내부평가 test dataset으로 시스템에 제출하였다.

데이터셋은 https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=180&clCd=END_TASK&subMenuId=sub01 에서 신청해서 다운로드받는다. (현재 대회마감으로 다운로드 불가)

본 레포지토리의 모든 코드는 아래의 데이터의 경로를 기반으로 작성되었다.

```
data/QA/
 ├── korean_culture_qa_V1.0_dev+.json
 ├── korean_culture_qa_V1.0_test+.json
 └── korean_culture_qa_V1.0_train+.json
```

만일 다른 경로에 데이터가 저장되었을 시, python 실행 시 argument로 입력하거나 .sh 파일의 각 데이터셋 경로를 수정한다.

# Code Usage

### Proposed Methodology(train)
Midm+CoT+QLoRA+GRPO+(B:cand_max)를 이용한 학습
```
bash GRPO_train.sh
```
or
```
CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --model_name "K-intelligence/Midm-2.0-Base-Instruct" \
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
```

- rl_mode "ppo" : PPO clipped surrogate loss를 이용한 GRPO적용
- use_format_reward : ormat reward 적용
- use_write_type_answer : Descriptive Answer Candidate 적용
- wta_reward_stretegy "cand_max" : (A): None (B): cand_max (C):1 (D):cand_include 중 택 1
- no_flash_attention : flash attention 사용하지 않을 시

### Proposed Methodology(reproduce test)

Midm의 시스템 프롬프트의 날짜정보를 하드코딩하여 완전재현 목적 
```
bash reproduce_test.sh 
# bash reproduce_test_no_flash_attn.sh #flash attention 환경 조성이 힘들거나 bash reproduce_test.sh 시 score가 재현이 안됐을시 시도
```
or
```
CUDA_VISIBLE_DEVICES=0 python -u reproduce_test.py \
    --adapter_path "GyunYeop/midm-base-GRPO-tuning-KoreanCultureQA" \
    --device "cuda:0"
```

### Proposed Methodology(general test)
일반 test
```
bash test.sh
```
or
```
bash test_no_flash_attn.sh
```
or
```
CUDA_VISIBLE_DEVICES=0 python -u test.py \
    --adapter_path "GyunYeop/midm-base-GRPO-tuning-KoreanCultureQA" \
    --num_beams 5 \
    --device "cuda:0"
```

### fine-tuning(비교군)
```
bash LoRA_train.sh
```
or
```
CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --model_name "K-intelligence/Midm-2.0-Base-Instruct" \
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
```

### 각 언어모델 비교하여 지식 량 평가 (모델기술서 표 3)
사용된 언어모델: 
```
bash find_models.sh
python calculate_expanded_measurement.py --target_folder find_models
```
 
# Methodology

<img width="6082" height="3248" alt="말평_2025_QA" src="https://github.com/user-attachments/assets/24b10e34-a6c0-40b3-bc95-335d5fdb7de2" />
CoT의 학습을 위한 GRPO와 RLVR학습

<img width="6099" height="2084" alt="그림2" src="https://github.com/user-attachments/assets/6e90d07e-8719-4447-b8b5-417769b1bcf4" />
Descriptive Answer Candidate를 이용한 서술형 답안의 직접학습

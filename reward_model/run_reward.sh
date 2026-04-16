#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 1. 关键：把项目根目录加入 PYTHONPATH
# 这样 Python 就能识别 from reward_model.rm_dataset
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_ENDPOINT=https://hf-mirror.com
# 配置路径
MODEL_PATH="Qwen/Qwen2.5-7B"  # 或者 1.5B 全参数
SAVE_PATH="/root/autodl-tmp/output/reward_model_v1"
mkdir -p $SAVE_PATH

# 启动训练
# 5090 跑 7B LoRA，可以将 Batch Size 设为 2，开启梯度检查点
python reward_model/train.py \
    --model_name "$MODEL_PATH" \
    --train_path "HuggingFaceH4/ultrafeedback_binarized" \
    --save_path "$SAVE_PATH" \
    --num_epochs 5 \
    --batch_size 4 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --max_length 2048 \
    --weight_decay 0.01 \
    --patience 3 \
    --seed 42 \
    --lora_r 64 \
    --lora_alpha 128 \

echo "Reward Model Training Finished!"
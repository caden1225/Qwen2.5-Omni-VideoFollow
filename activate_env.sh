#!/bin/bash

# 激活conda环境
echo "正在激活conda环境 312..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 312

# 设置环境变量
export MODEL_PATH="/home/caden/workplace/models/Qwen2.5-Omni-3B"
export CONDA_ENV="312"

echo "环境已激活！"
echo "当前conda环境: $CONDA_ENV"
echo "模型路径: $MODEL_PATH"

# 显示当前Python路径
echo "Python路径: $(which python)"
echo "Python版本: $(python --version)"

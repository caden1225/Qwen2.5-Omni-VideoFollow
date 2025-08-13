#!/bin/bash

# 运行Qwen2.5-Omni推理的脚本

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

# 检查参数
if [ $# -eq 0 ]; then
    echo "使用方法: $0 [hf_infer|hf_infer_video_text]"
    echo "  hf_infer: 运行完整的音频+视频推理"
    echo "  hf_infer_video_text: 运行仅视频+文本推理"
    exit 1
fi

SCRIPT_NAME=$1

case $SCRIPT_NAME in
    "hf_infer")
        echo "运行完整的音频+视频推理..."
        python hf_infer.py
        ;;
    "hf_infer_video_text")
        echo "运行仅视频+文本推理..."
        python hf_infer_video_text.py
        ;;
    *)
        echo "未知的脚本名称: $SCRIPT_NAME"
        echo "支持: hf_infer, hf_infer_video_text"
        exit 1
        ;;
esac

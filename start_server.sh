#!/bin/bash

# Nanonets-OCR-s vLLM 服务器启动脚本
# 使用conda环境312

echo "=== 启动 Nanonets-OCR-s vLLM 服务器 ==="

# 检查conda环境
if ! conda env list | grep -q "312"; then
    echo "错误: 未找到conda环境312"
    echo "请先创建conda环境312: conda create -n 312 python=3.12"
    exit 1
fi

# 检查模型路径
MODEL_PATH="/home/caden/workplace/nanonets/Nanonets-OCR-s"
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

echo "✓ 找到conda环境312"
echo "✓ 模型路径: $MODEL_PATH"

# 安装vLLM（如果需要）
echo "检查vLLM安装..."
if ! conda run -n 312 python -c "import vllm" 2>/dev/null; then
    echo "正在安装vLLM..."
    conda run -n 312 pip install vllm[all]
fi

echo "✓ vLLM已安装"

# 启动服务器
echo "正在启动vLLM服务器..."
echo "服务器地址: http://localhost:8000"
echo "按Ctrl+C停止服务器"
echo ""

conda run -n 312 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 
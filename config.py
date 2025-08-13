"""
Qwen2.5-Omni 项目配置文件
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 模型配置
MODEL_PATH = os.getenv('MODEL_PATH', "/home/caden/workplace/models/Qwen2.5-Omni-3B")

# 环境配置
CONDA_ENV = os.getenv('CONDA_ENV', '312')

# qwen-omni-utils 路径
QWEN_OMNI_UTILS_PATH = PROJECT_ROOT / "qwen-omni-utils" / "src"

# 其他配置
USE_AUDIO_IN_VIDEO = True  # 是否在视频中处理音频
MAX_NEW_TOKENS = 256  # 最大生成token数
TEMPERATURE = 0.7  # 生成温度
TOP_P = 0.9  # top-p采样参数

# 内存优化配置
PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

# Qwen2.5-Omni 项目设置说明

## 项目结构

```
qwen2.5-Omni_inference/
├── qwen-omni-utils/          # 项目本地的qwen-omni-utils
├── hf_infer.py               # 完整的音频+视频推理脚本
├── hf_infer_video_text.py   # 仅视频+文本推理脚本
├── config.py                 # 配置文件
├── activate_env.sh           # 环境激活脚本
├── run_inference.sh          # 推理运行脚本
└── PROJECT_SETUP.md          # 本说明文档
```

## 环境要求

- Python 3.8+
- Conda 环境 312
- CUDA 支持的 GPU
- 足够的 GPU 内存（建议 8GB+）

## 快速开始

### 1. 激活环境

```bash
# 方法1: 使用激活脚本
./activate_env.sh

# 方法2: 手动激活
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 312
```

### 2. 设置环境变量

```bash
export MODEL_PATH="/home/caden/workplace/models/Qwen2.5-Omni-3B"
export CONDA_ENV="312"
```

### 3. 运行推理

```bash
# 运行完整的音频+视频推理
./run_inference.sh hf_infer

# 运行仅视频+文本推理
./run_inference.sh hf_infer_video_text

# 或者直接运行Python脚本
python hf_infer.py
python hf_infer_video_text.py
```

## 配置说明

### 模型路径配置

模型路径可以通过以下方式设置：

1. **环境变量**（推荐）：
   ```bash
   export MODEL_PATH="/path/to/your/model"
   ```

2. **默认路径**：
   如果没有设置环境变量，将使用默认路径：
   `/home/caden/workplace/models/Qwen2.5-Omni-3B`

### 其他配置

在 `config.py` 文件中可以调整以下参数：

- `USE_AUDIO_IN_VIDEO`: 是否在视频中处理音频
- `MAX_NEW_TOKENS`: 最大生成token数
- `TEMPERATURE`: 生成温度
- `TOP_P`: top-p采样参数

## 主要改进

1. **本地依赖**: 使用项目目录中的 `qwen-omni-utils` 而不是外部安装
2. **环境变量支持**: 通过环境变量配置模型路径
3. **路径管理**: 自动添加正确的Python路径
4. **脚本化**: 提供便捷的运行脚本
5. **配置集中化**: 所有配置集中在 `config.py` 文件中

## 故障排除

### 导入错误

如果遇到 `qwen_omni_utils` 导入错误，请检查：

1. 确保 `qwen-omni-utils/` 目录存在
2. 确保已激活正确的conda环境
3. 检查Python路径设置

### 内存不足

如果遇到GPU内存不足：

1. 减少 `MAX_NEW_TOKENS` 值
2. 设置 `USE_AUDIO_IN_VIDEO = False`
3. 使用更小的模型版本

### 模型路径错误

如果模型加载失败：

1. 检查 `MODEL_PATH` 环境变量
2. 确认模型文件存在且完整
3. 检查文件权限

## 注意事项

- 首次运行可能需要下载模型文件
- 确保有足够的磁盘空间存储模型
- 建议在运行前清理GPU内存
- 长时间运行建议使用 `nohup` 或 `screen`

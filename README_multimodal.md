# Qwen2.5-Omni 多模态应用

## 🎯 项目概述

本项目基于Qwen2.5-Omni模型，提供了完整的多模态AI应用解决方案，支持文本、图像、音频、视频等多种输入模态的组合处理。

## 📋 内存溢出问题分析

### low_VRAM_demo_3B_fixed.py 内存溢出原因：

1. **设备映射不当**: 模型的视觉和音频组件默认加载到GPU，占用大量显存
2. **批处理**: 同时处理多个模态数据时显存需求激增
3. **Flash Attention**: 使用flash_attention_2时对显存要求更高
4. **模型组件**: thinker.visual和thinker.audio_tower组件较大

### 解决方案：

- 使用`device_map`将视觉和音频组件放到CPU
- 采用低显存模式的建模方式
- 合理配置torch_dtype为float16
- 及时清理GPU缓存

## 🚀 功能特性

### 1. 多模态输入支持
- 📝 **文本**: 纯文本对话和指令
- 🖼️ **图像**: 支持各种格式的图像输入
- 🎵 **音频**: 音频理解和处理
- 🎬 **视频**: 完整视频内容分析

### 2. 视频特殊处理
- 📢 **音轨提取**: 自动提取视频中的音频内容
- 🖼️ **帧提取**: 提取视频最后一帧作为图像输入
- 🎯 **智能转换**: 视频→音频+图像的组合处理

### 3. 多种部署方式
- 🔥 **API服务**: FastAPI构建的RESTful API
- 🎨 **Web界面**: Gradio构建的用户友好界面
- ⚡ **低显存模式**: 支持显存受限的环境

## 📦 安装依赖

```bash
# 安装多模态应用依赖
pip install -r requirements_multimodal.txt

# 安装qwen-omni-utils
cd qwen-omni-utils
pip install -e .
```

## ⚙️ 环境配置

创建 `.env` 文件：

```bash
MODEL_PATH="/home/caden/models/Qwen2.5-Omni-3B"
```

## 🖥️ 使用方法

### 1. 启动API服务

```bash
python multimodal_api.py
```

API将在 `http://localhost:8000` 启动，支持以下端点：
- `POST /multimodal` - 多模态推理
- `GET /health` - 健康检查
- `GET /` - API信息

### 2. 启动Gradio界面

```bash
python gradio_multimodal_app.py
```

Web界面将在 `http://localhost:7860` 启动

### 3. API使用示例

```python
import requests

# 文本+图像输入
files = {
    'images': open('image.jpg', 'rb')
}
data = {
    'text': '描述这张图片',
    'system_prompt': 'You are a helpful AI assistant.',
    'max_new_tokens': 512
}

response = requests.post('http://localhost:8000/multimodal', files=files, data=data)
print(response.json())
```

## 🎛️ 配置参数

### API参数

- `text`: 文本输入 (可选)
- `system_prompt`: 系统提示 (默认: "You are a helpful AI assistant.")
- `max_new_tokens`: 最大生成token数 (默认: 512)
- `extract_video_audio`: 是否提取视频音轨 (默认: False)
- `extract_video_frame`: 是否提取视频帧 (默认: False)

### 文件上传

- `images`: 图像文件列表
- `audios`: 音频文件列表  
- `videos`: 视频文件列表

## 📊 性能优化

### 显存优化策略

1. **设备映射**: 将大型组件分配到CPU
```python
device_map = {
    "thinker.model": "cuda",
    "thinker.lm_head": "cuda", 
    "thinker.visual": "cpu",
    "thinker.audio_tower": "cpu",
    "talker": "cuda",
    "token2wav": "cuda",
}
```

2. **数据类型**: 使用float16减少显存占用
3. **批处理**: 控制同时处理的数据量
4. **缓存管理**: 及时清理GPU缓存

## 🔧 故障排除

### 常见问题

1. **内存不足**: 
   - 检查设备映射配置
   - 减少max_new_tokens
   - 使用更小的输入分辨率

2. **模型加载失败**:
   - 检查MODEL_PATH环境变量
   - 确认模型文件完整性
   - 检查依赖版本兼容性

3. **处理超时**:
   - 检查输入数据大小
   - 调整处理超时时间
   - 确认硬件性能

## 📁 文件结构

```
├── multimodal_api.py          # FastAPI服务
├── gradio_multimodal_app.py   # Gradio界面
├── requirements_multimodal.txt # 依赖列表
├── low-VRAM-mode/            # 低显存模式
│   ├── low_VRAM_demo_3B_fixed.py
│   └── modeling_qwen2_5_omni_low_VRAM_mode.py
└── qwen-omni-utils/          # 预处理工具
    └── src/qwen_omni_utils/
        ├── __init__.py
        └── v2_5/
            ├── audio_process.py
            └── vision_process.py
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

本项目遵循原项目的许可证条款。
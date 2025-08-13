# 🎥 多模态视频处理服务

一个支持多种输入方式的多模态内容处理服务，包括视频、音频+图片、文本+图片等输入方式，输出为文本描述和分析结果。

## ✨ 功能特性

### 🎬 视频文件处理
- 自动提取音频轨道
- 提取关键帧图像
- 支持多种视频格式（MP4, AVI, MOV, MKV, WMV）
- 智能帧选择和时间间隔控制

### 🎵 音频+图片处理
- 音频特征分析（时长、能量、频谱质心等）
- 图片预处理和优化
- 多模态内容融合分析

### 📝 文本+图片处理
- 文本内容分析
- 图片内容识别
- 图文关联分析

### 🌐 Web界面
- 美观的Gradio界面
- 实时处理状态显示
- 结果下载和导出
- 响应式设计

## 🚀 快速开始

### 环境要求
- Python 3.8+
- CUDA支持（可选，用于GPU加速）
- FFmpeg（用于音频处理）

### 安装依赖

```bash
# 激活conda环境（推荐使用Python 3.12）
conda activate 312

# 安装依赖
pip install -r requirements_multimodal.txt
```

### 启动服务

```bash
# 方式1: 使用启动脚本
python start_service.py

# 方式2: 直接启动Gradio界面
python gradio_multimodal_interface.py

# 方式3: 测试服务功能
python test_multimodal_service.py
```

服务启动后，在浏览器中访问 `http://localhost:7860` 即可使用界面。

## 📁 项目结构

```
├── multimodal_video_service.py      # 核心服务模块
├── gradio_multimodal_interface.py   # Gradio界面
├── start_service.py                 # 启动脚本
├── test_multimodal_service.py       # 测试脚本
├── requirements_multimodal.txt      # 依赖文件
└── README_multimodal_service.md     # 说明文档
```

## 🔧 配置选项

### 处理配置
- **音频采样率**: 默认16000Hz
- **图片最大尺寸**: 默认1024px
- **视频最大帧数**: 默认8帧
- **帧提取间隔**: 默认1秒

### 输出配置
- **输出格式**: 文本描述
- **最大输出长度**: 1000字符
- **保存中间文件**: 可选

## 📖 使用说明

### 1. 视频文件处理
1. 选择"视频文件"输入方式
2. 上传视频文件
3. 点击"处理视频"按钮
4. 等待处理完成，查看结果

### 2. 音频+图片处理
1. 选择"音频+图片"输入方式
2. 上传音频文件和图片
3. 点击"处理音频+图片"按钮
4. 查看分析结果

### 3. 文本+图片处理
1. 选择"文本+图片"输入方式
2. 输入文本内容并上传图片
3. 点击"处理文本+图片"按钮
4. 获取分析结果

## 🔍 API接口

### 核心服务类
```python
from multimodal_video_service import MultimodalVideoService

# 创建服务实例
service = MultimodalVideoService()

# 处理视频
result = service.process_input(video_file="video.mp4")

# 处理音频+图片
result = service.process_input(
    audio_file="audio.wav", 
    image_file="image.jpg"
)

# 处理文本+图片
result = service.process_input(
    text_input="分析文本", 
    image_file="image.jpg"
)
```

### 直接调用函数
```python
from multimodal_video_service import process_multimodal_input

# 处理多模态输入
result = process_multimodal_input(
    video_file="video.mp4",
    audio_file="audio.wav",
    image_file="image.jpg",
    text_input="分析文本"
)
```

## 🧪 测试

运行测试脚本验证服务功能：

```bash
python test_multimodal_service.py
```

测试内容包括：
- 核心服务功能测试
- 各种输入方式测试
- Gradio界面测试
- 错误处理测试

## 🐛 故障排除

### 常见问题

1. **依赖安装失败**
   ```bash
   # 升级pip
   pip install --upgrade pip
   
   # 使用conda安装
   conda install pytorch torchvision torchaudio -c pytorch
   ```

2. **FFmpeg未找到**
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # CentOS/RHEL
   sudo yum install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

3. **CUDA内存不足**
   ```bash
   # 设置环境变量
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

4. **端口被占用**
   ```bash
   # 修改端口
   python gradio_multimodal_interface.py --port 7861
   ```

### 日志查看
服务运行时会输出详细的日志信息，包括：
- 处理状态
- 错误信息
- 性能指标
- 调试信息

## 🔮 扩展功能

### 计划中的功能
- [ ] 支持更多音频格式
- [ ] 视频质量评估
- [ ] 批量处理支持
- [ ] 结果缓存机制
- [ ] 用户认证系统

### 自定义扩展
可以通过继承基类来扩展功能：

```python
class CustomVideoProcessor(MultimodalVideoProcessor):
    def custom_processing(self, input_data):
        # 自定义处理逻辑
        pass
```

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📞 支持

如有问题或建议，请：
1. 查看本文档
2. 搜索现有Issue
3. 创建新的Issue
4. 联系项目维护者

---

**享受多模态内容处理的乐趣！** 🎉

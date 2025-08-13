# 音轨和视频分离处理功能实现

## 概述

本项目成功实现了音轨和视频分离处理功能，可以从视频文件中分别提取音频和关键帧图像，为后续的多模态AI处理提供更灵活的数据输入方式。

## 功能特性

### 🎵 音频提取
- 支持从MP4等视频格式中提取音频
- 可配置采样率（默认16000Hz）
- 支持多种音频格式输出（WAV等）
- 自动检测视频是否包含音频轨道

### 🖼️ 关键帧提取
- 支持多种提取策略：
  - `last`: 提取最后一帧
  - `keyframes`: 均匀分布的关键帧
  - `uniform`: 均匀分布的帧
  - `custom`: 自定义帧索引
- 可配置关键帧数量
- 自动调整图像质量（JPEG 1-100）
- 保存时包含时间戳信息

### 🎬 视频处理
- 可选的视频张量处理
- 支持帧数、分辨率、时间范围等参数配置
- 内存优化（半精度、像素限制等）

## 文件结构

```
├── test_audio_video_separation.py          # 基础分离功能测试
├── enhanced_video_processor.py             # 增强版视频处理模块
├── test_enhanced_video_processor.py        # 增强版模块测试
├── extracted_media/                        # 提取的媒体文件
│   ├── audio/                             # 音频文件
│   ├── frames/                            # 关键帧图像
│   └── video/                             # 视频文件（预留）
└── README_audio_video_separation.md        # 本文档
```

## 预设配置

### 1. audio_focus（音频专注）
- 音频提取：✅ 启用
- 帧提取：✅ 启用（2帧关键帧）
- 视频处理：❌ 禁用
- 适用场景：主要关注音频内容的分析

### 2. frame_focus（帧专注）
- 音频提取：✅ 启用
- 帧提取：✅ 启用（5帧均匀分布）
- 视频处理：✅ 启用
- 适用场景：主要关注视觉内容的分析

### 3. balanced_separation（平衡分离）
- 音频提取：✅ 启用
- 帧提取：✅ 启用（3帧关键帧）
- 视频处理：✅ 启用
- 适用场景：平衡音频和视觉内容分析

### 4. full_extraction（完整提取）
- 音频提取：✅ 启用
- 帧提取：✅ 启用（8帧均匀分布）
- 视频处理：✅ 启用
- 适用场景：需要完整的多模态分析

## 使用方法

### 基础使用

```python
from enhanced_video_processor import (
    EnhancedVideoProcessor,
    VideoOptimizationConfig,
    AudioVideoSeparationConfig
)

# 创建配置
video_config = VideoOptimizationConfig(
    nframes=4,
    resized_height=168,
    resized_width=168
)

separation_config = AudioVideoSeparationConfig(
    extract_audio=True,
    extract_frames=True,
    frame_extraction_method="keyframes",
    num_keyframes=3
)

# 创建处理器
processor = EnhancedVideoProcessor(video_config, separation_config)

# 处理视频
success, results, media_data = processor.process_video_with_separation(
    "video.mp4", 
    conversation
)
```

### 使用预设配置

```python
from enhanced_video_processor import EnhancedVideoOptimizationPresets

# 获取预设配置
video_config, separation_config = EnhancedVideoOptimizationPresets.get_separation_preset('balanced_separation')

# 创建处理器
processor = EnhancedVideoProcessor(video_config, separation_config)
```

## 测试结果

### 测试视频：math.mp4
- 文件大小：82.76 MB
- 分辨率：3840x2160 (4K)
- 时长：30.37秒
- 帧率：30 FPS
- 音频：有（44100Hz）

### 处理结果
| 预设配置 | 音频提取 | 帧提取 | 处理时间 | 推荐度 |
|---------|---------|--------|----------|--------|
| audio_focus | ✅ | ✅ (2帧) | 3.17秒 | ⭐⭐⭐ |
| frame_focus | ✅ | ✅ (5帧) | 3.46秒 | ⭐⭐⭐⭐ |
| balanced_separation | ✅ | ✅ (3帧) | 2.43秒 | ⭐⭐⭐⭐⭐ |
| full_extraction | ✅ | ✅ (8帧) | 6.08秒 | ⭐⭐⭐⭐ |

## Processor能力测试

### ✅ 支持的功能
1. **纯文本输入** - 基础文本处理
2. **图像+文本输入** - 视觉内容分析
3. **音频+文本输入** - 音频内容分析
4. **视频+文本输入** - 视频内容分析
5. **混合输入** - 图像+音频+文本组合

### ⚠️ 注意事项
- 混合输入处理时需要注意张量格式
- 音频输出功能需要默认系统提示词
- 视频处理功能在某些配置下可能失败

## 技术实现

### 核心依赖
- **OpenCV (cv2)**: 视频帧提取和图像处理
- **librosa**: 音频提取和处理
- **soundfile**: 音频文件保存
- **PIL/Pillow**: 图像处理
- **torch**: 深度学习张量处理
- **transformers**: Qwen2.5-Omni模型支持

### 关键算法
1. **音频提取**: 使用librosa.load()从视频中提取音频数据
2. **帧提取**: 使用cv2.VideoCapture()按策略提取关键帧
3. **智能调整**: 根据配置自动调整分辨率、帧数等参数
4. **内存优化**: 支持半精度、像素限制等优化策略

## 扩展功能

### 未来改进方向
1. **更多音频格式支持**: MP3, AAC, FLAC等
2. **高级帧提取算法**: 基于内容的关键帧检测
3. **视频压缩优化**: 更智能的压缩策略
4. **批量处理**: 支持多个视频文件并行处理
5. **GPU加速**: 利用GPU加速视频处理

### 配置持久化
- 支持从JSON文件加载/保存配置
- 可配置的输出目录和文件命名规则
- 中间文件清理选项

## 总结

本项目成功实现了音轨和视频分离处理功能，主要特点包括：

1. **功能完整**: 支持音频提取、关键帧提取、视频处理
2. **配置灵活**: 提供多种预设配置，满足不同使用场景
3. **性能优化**: 处理时间短，内存占用合理
4. **扩展性强**: 模块化设计，易于扩展和维护
5. **测试充分**: 包含完整的测试用例和性能分析

通过这种分离处理方式，我们可以：
- 更灵活地处理多模态数据
- 减少内存占用和计算复杂度
- 支持不同的AI分析需求
- 为后续的多模态AI应用提供更好的数据基础

推荐使用 `balanced_separation` 预设配置，它在处理时间和功能完整性之间取得了最佳平衡。

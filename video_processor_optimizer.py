#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理优化模块
包含所有验证成功的抽样、降帧、调整功能
"""

import torch
import gc
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "qwen-omni-utils" / "src"))

from qwen_omni_utils import process_mm_info

@dataclass
class VideoOptimizationConfig:
    """视频优化配置类"""
    # 帧数控制
    nframes: int = 4                    # 直接指定帧数
    fps: Optional[float] = None         # 帧率（与nframes互斥）
    min_frames: int = 2                 # 最小帧数
    max_frames: int = 16                # 最大帧数
    
    # 分辨率控制
    resized_height: int = 112           # 目标高度
    resized_width: int = 112            # 目标宽度
    
    # 时间控制
    video_start: float = 0.0            # 开始时间（秒）
    video_end: Optional[float] = None   # 结束时间（秒）
    
    # 像素限制
    min_pixels: Optional[int] = None    # 最小像素数
    max_pixels: Optional[int] = None    # 最大像素数
    
    # 内存优化
    use_half_precision: bool = True     # 是否使用float16
    enable_audio: bool = False          # 是否启用音频处理

class VideoProcessorOptimizer:
    """视频处理优化器"""
    
    def __init__(self, config: VideoOptimizationConfig):
        self.config = config
        self.setup_environment()
    
    def setup_environment(self):
        """设置环境变量"""
        # 设置视频像素限制
        if self.config.max_pixels:
            os.environ['VIDEO_MAX_PIXELS'] = str(self.config.max_pixels)
        
        # 设置CUDA内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    def validate_config(self) -> bool:
        """验证配置参数"""
        # 检查fps和nframes不能同时使用
        if self.config.fps is not None and self.config.nframes is not None:
            print("❌ 错误：fps和nframes不能同时使用")
            return False
        
        # 检查分辨率是否合理
        if self.config.resized_height < 112 or self.config.resized_width < 112:
            print("❌ 错误：分辨率不能小于112x112")
            return False
        
        # 检查时间范围
        if self.config.video_end and self.config.video_start >= self.config.video_end:
            print("❌ 错误：开始时间必须小于结束时间")
            return False
        
        return True
    
    def get_video_params(self) -> Dict[str, Any]:
        """获取视频处理参数"""
        params = {}
        
        # 帧数控制
        if self.config.nframes is not None:
            params['nframes'] = self.config.nframes
        elif self.config.fps is not None:
            params['fps'] = self.config.fps
            params['min_frames'] = self.config.min_frames
            params['max_frames'] = self.config.max_frames
        
        # 分辨率控制
        params['resized_height'] = self.config.resized_height
        params['resized_width'] = self.config.resized_width
        
        # 时间控制
        if self.config.video_start > 0:
            params['video_start'] = self.config.video_start
        if self.config.video_end:
            params['video_end'] = self.config.video_end
        
        # 像素限制
        if self.config.min_pixels:
            params['min_pixels'] = self.config.min_pixels
        if self.config.max_pixels:
            params['max_pixels'] = self.config.max_pixels
        
        return params
    
    def process_video(self, video_path: str, conversation: list) -> Tuple[bool, Optional[torch.Tensor], Dict[str, Any]]:
        """
        处理视频
        
        Args:
            video_path: 视频文件路径
            conversation: 对话内容
            
        Returns:
            (成功标志, 视频张量, 处理信息)
        """
        if not self.validate_config():
            return False, None, {}
        
        try:
            # 检查文件
            if not os.path.exists(video_path):
                print(f"❌ 视频文件不存在: {video_path}")
                return False, None, {}
            
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            print(f"📹 处理视频: {video_path}")
            print(f"📊 文件大小: {file_size:.2f} MB")
            
            # 获取处理参数
            video_params = self.get_video_params()
            print(f"⚙️ 处理参数: {video_params}")
            
            # 创建带视频的对话
            video_conversation = self._add_video_to_conversation(conversation, video_path, video_params)
            
            # 处理多媒体信息
            start_time = time.time()
            audios, images, videos = process_mm_info(video_conversation, use_audio_in_video=self.config.enable_audio)
            processing_time = time.time() - start_time
            
            if not videos:
                print("❌ 视频处理失败")
                return False, None, {}
            
            video_tensor = videos[0]
            print(f"✅ 视频处理成功")
            print(f"  - 形状: {video_tensor.shape}")
            print(f"  - 内存占用: {video_tensor.element_size() * video_tensor.nelement() / 1024**2:.2f} MB")
            print(f"  - 处理时间: {processing_time:.2f}秒")
            
            # 转换为半精度以节省内存
            if self.config.use_half_precision and video_tensor.dtype == torch.float32:
                video_tensor = video_tensor.half()
                print(f"  - 转换为float16后内存: {video_tensor.element_size() * video_tensor.nelement() / 1024**2:.2f} MB")
            
            # 收集处理信息
            info = {
                'file_size_mb': file_size,
                'processing_time': processing_time,
                'final_shape': list(video_tensor.shape),
                'final_memory_mb': video_tensor.element_size() * video_tensor.nelement() / 1024**2,
                'params_used': video_params
            }
            
            return True, video_tensor, info
            
        except Exception as e:
            print(f"❌ 视频处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False, None, {}
    
    def _add_video_to_conversation(self, conversation: list, video_path: str, video_params: Dict[str, Any]) -> list:
        """将视频添加到对话中"""
        # 深拷贝对话
        video_conversation = []
        for turn in conversation:
            new_turn = {'role': turn['role'], 'content': []}
            for content in turn['content']:
                if content['type'] == 'video':
                    # 替换视频内容
                    new_content = {'type': 'video', 'video': video_path, **video_params}
                    new_turn['content'].append(new_content)
                else:
                    new_turn['content'].append(content.copy())
            video_conversation.append(new_turn)
        
        return video_conversation
    
    def cleanup(self):
        """清理资源"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class VideoOptimizationPresets:
    """视频优化预设配置"""
    
    @staticmethod
    def get_preset(name: str) -> VideoOptimizationConfig:
        """获取预设配置"""
        presets = {
            'extreme_low_memory': VideoOptimizationConfig(
                nframes=2,
                resized_height=112,
                resized_width=112,
                video_start=0.0,
                video_end=2.0,
                max_pixels=64 * 28 * 28,
                use_half_precision=True,
                enable_audio=False
            ),
            
            'low_memory': VideoOptimizationConfig(
                nframes=4,
                resized_height=112,
                resized_width=112,
                video_start=0.0,
                video_end=3.0,
                max_pixels=128 * 28 * 28,
                use_half_precision=True,
                enable_audio=False
            ),
            
            'balanced': VideoOptimizationConfig(
                nframes=6,
                resized_height=168,
                resized_width=168,
                video_start=0.0,
                video_end=4.0,
                max_pixels=256 * 28 * 28,
                use_half_precision=True,
                enable_audio=False
            ),
            
            'high_quality': VideoOptimizationConfig(
                nframes=8,
                resized_height=224,
                resized_width=224,
                video_start=0.0,
                video_end=5.0,
                max_pixels=512 * 28 * 28,
                use_half_precision=True,
                enable_audio=False
            ),
            
            'custom_math_video': VideoOptimizationConfig(
                nframes=4,
                resized_height=112,
                resized_width=112,
                video_start=0.0,
                video_end=3.0,
                max_pixels=128 * 28 * 28,
                use_half_precision=True,
                enable_audio=False
            )
        }
        
        if name not in presets:
            raise ValueError(f"未知的预设配置: {name}。可用配置: {list(presets.keys())}")
        
        return presets[name]
    
    @staticmethod
    def list_presets() -> list:
        """列出所有可用的预设配置"""
        return [
            'extreme_low_memory',  # 极低内存模式
            'low_memory',          # 低内存模式
            'balanced',            # 平衡模式
            'high_quality',        # 高质量模式
            'custom_math_video'    # 数学视频专用模式
        ]

def load_config_from_file(config_path: str) -> VideoOptimizationConfig:
    """从配置文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 创建配置对象
        config = VideoOptimizationConfig(**config_dict)
        return config
        
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        # 返回默认配置
        return VideoOptimizationConfig()

def save_config_to_file(config: VideoOptimizationConfig, config_path: str):
    """保存配置到文件"""
    try:
        config_dict = {
            'nframes': config.nframes,
            'fps': config.fps,
            'min_frames': config.min_frames,
            'max_frames': config.max_frames,
            'resized_height': config.resized_height,
            'resized_width': config.resized_width,
            'video_start': config.video_start,
            'video_end': config.video_end,
            'min_pixels': config.min_pixels,
            'max_pixels': config.max_pixels,
            'use_half_precision': config.use_half_precision,
            'enable_audio': config.enable_audio
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 配置已保存到: {config_path}")
        
    except Exception as e:
        print(f"❌ 保存配置文件失败: {e}")

if __name__ == "__main__":
    # 测试预设配置
    print("=== 视频优化预设配置测试 ===")
    
    for preset_name in VideoOptimizationPresets.list_presets():
        print(f"\n--- 预设: {preset_name} ---")
        config = VideoOptimizationPresets.get_preset(preset_name)
        print(f"配置: {config}")
        
        # 验证配置
        optimizer = VideoProcessorOptimizer(config)
        if optimizer.validate_config():
            print("✅ 配置验证通过")
        else:
            print("❌ 配置验证失败")
    
    # 保存示例配置
    example_config = VideoOptimizationPresets.get_preset('low_memory')
    save_config_to_file(example_config, 'example_video_config.json')

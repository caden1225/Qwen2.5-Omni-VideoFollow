#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版视频处理模块
支持音轨和视频分离处理，提取音频和关键帧
"""

import torch
import gc
import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import librosa
from PIL import Image

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "qwen-omni-utils" / "src"))

from qwen_omni_utils import process_mm_info

@dataclass
class AudioVideoSeparationConfig:
    """音视频分离处理配置"""
    # 音频提取配置
    extract_audio: bool = True              # 是否提取音频
    audio_sample_rate: int = 16000         # 音频采样率
    audio_format: str = "wav"              # 音频格式
    audio_quality: str = "high"            # 音频质量 (low, medium, high)
    
    # 图像提取配置
    extract_frames: bool = True             # 是否提取关键帧
    frame_extraction_method: str = "last"  # 帧提取方法 (last, keyframes, uniform, custom)
    num_keyframes: int = 3                 # 关键帧数量
    frame_quality: int = 95                # JPEG质量 (1-100)
    
    # 视频处理配置
    video_processing: bool = True           # 是否处理视频
    video_compression: bool = True          # 是否压缩视频
    video_quality: str = "medium"          # 视频质量 (low, medium, high)
    
    # 输出配置
    output_dir: str = "./extracted_media"  # 输出目录
    save_intermediate: bool = True         # 是否保存中间文件
    cleanup_after_processing: bool = False # 处理后是否清理中间文件

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

class EnhancedVideoProcessor:
    """增强版视频处理器，支持音视频分离"""
    
    def __init__(self, 
                 video_config: VideoOptimizationConfig,
                 separation_config: AudioVideoSeparationConfig):
        self.video_config = video_config
        self.separation_config = separation_config
        self.setup_environment()
        self.setup_output_directory()
    
    def setup_environment(self):
        """设置环境变量"""
        # 设置视频像素限制
        if self.video_config.max_pixels:
            os.environ['VIDEO_MAX_PIXELS'] = str(self.video_config.max_pixels)
        
        # 设置CUDA内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    def setup_output_directory(self):
        """设置输出目录"""
        output_dir = Path(self.separation_config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (output_dir / "audio").mkdir(exist_ok=True)
        (output_dir / "frames").mkdir(exist_ok=True)
        (output_dir / "video").mkdir(exist_ok=True)
        
        print(f"📁 输出目录已创建: {output_dir.absolute()}")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """获取视频基本信息"""
        print(f"📊 分析视频信息: {os.path.basename(video_path)}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")
            
            # 基本信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # 检查音频轨道
            has_audio = False
            try:
                audio, sr = librosa.load(video_path, sr=None)
                has_audio = True
                audio_duration = len(audio) / sr
            except:
                has_audio = False
                audio_duration = 0
            
            cap.release()
            
            info = {
                'file_path': video_path,
                'file_size_mb': os.path.getsize(video_path) / (1024 * 1024),
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration': duration,
                'has_audio': has_audio,
                'audio_duration': audio_duration,
                'resolution': f"{width}x{height}",
                'aspect_ratio': width / height if height > 0 else 0
            }
            
            print(f"  ✅ 视频信息获取成功:")
            print(f"    📐 分辨率: {info['resolution']}")
            print(f"    🎬 总帧数: {total_frames}")
            print(f"    ⏱️ 时长: {duration:.2f}秒")
            print(f"    🎵 音频: {'有' if has_audio else '无'}")
            if has_audio:
                print(f"    🔊 音频时长: {audio_duration:.2f}秒")
            
            return info
            
        except Exception as e:
            print(f"❌ 视频信息获取失败: {e}")
            return {}
    
    def extract_audio_from_video(self, video_path: str) -> Tuple[bool, Optional[str], Optional[np.ndarray]]:
        """从视频中提取音频"""
        if not self.separation_config.extract_audio:
            print("⚠️ 音频提取已禁用")
            return False, None, None
        
        print(f"🎵 提取音频: {os.path.basename(video_path)}")
        
        try:
            # 使用librosa提取音频
            audio, sr = librosa.load(video_path, sr=self.separation_config.audio_sample_rate)
            
            # 生成输出文件名
            video_name = Path(video_path).stem
            output_path = Path(self.separation_config.output_dir) / "audio" / f"{video_name}_audio.{self.separation_config.audio_format}"
            
            # 保存音频文件（使用soundfile，因为librosa.output已被移除）
            import soundfile as sf
            sf.write(str(output_path), audio, sr)
            
            print(f"  ✅ 音频提取成功:")
            print(f"    🎵 采样率: {sr} Hz")
            print(f"    🔊 音频长度: {len(audio)/sr:.2f}秒")
            print(f"    📊 音频形状: {audio.shape}")
            print(f"    💾 保存路径: {output_path}")
            
            return True, str(output_path), audio
            
        except Exception as e:
            print(f"  ❌ 音频提取失败: {e}")
            return False, None, None
    
    def extract_frames_from_video(self, video_path: str) -> Tuple[bool, List[str], List[np.ndarray]]:
        """从视频中提取关键帧"""
        if not self.separation_config.extract_frames:
            print("⚠️ 帧提取已禁用")
            return False, [], []
        
        print(f"🖼️ 提取关键帧: {os.path.basename(video_path)}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                raise ValueError("视频没有有效帧")
            
            extracted_frames = []
            frame_paths = []
            video_name = Path(video_path).stem
            
            if self.separation_config.frame_extraction_method == "last":
                # 提取最后一帧
                frame_indices = [total_frames - 1]
                
            elif self.separation_config.frame_extraction_method == "keyframes":
                # 提取关键帧（均匀分布）
                num_frames = min(self.separation_config.num_keyframes, total_frames)
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                
            elif self.separation_config.frame_extraction_method == "uniform":
                # 均匀提取帧
                num_frames = min(self.separation_config.num_keyframes, total_frames)
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                
            else:
                # 自定义帧索引
                frame_indices = [0, total_frames // 2, total_frames - 1]
            
            print(f"  📊 提取策略: {self.separation_config.frame_extraction_method}")
            print(f"  🎯 目标帧数: {len(frame_indices)}")
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # 转换BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 生成输出文件名
                    timestamp = frame_idx / fps if fps > 0 else 0
                    output_path = Path(self.separation_config.output_dir) / "frames" / f"{video_name}_frame_{i:02d}_{timestamp:.1f}s.jpg"
                    
                    # 保存图像
                    cv2.imwrite(str(output_path), frame, 
                               [cv2.IMWRITE_JPEG_QUALITY, self.separation_config.frame_quality])
                    
                    extracted_frames.append(frame_rgb)
                    frame_paths.append(str(output_path))
                    
                    print(f"    ✅ 帧 {i+1}: 时间 {timestamp:.1f}s, 保存到 {output_path.name}")
                else:
                    print(f"    ⚠️ 帧 {i+1}: 读取失败")
            
            cap.release()
            
            if extracted_frames:
                print(f"  🎉 帧提取完成: {len(extracted_frames)}/{len(frame_indices)} 帧成功")
                return True, frame_paths, extracted_frames
            else:
                print(f"  ❌ 没有成功提取任何帧")
                return False, [], []
                
        except Exception as e:
            print(f"  ❌ 帧提取失败: {e}")
            return False, [], []
    
    def process_video_with_separation(self, video_path: str, conversation: list) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """使用分离处理方式处理视频"""
        print(f"\n🎬 开始分离处理视频: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # 1. 获取视频信息
            video_info = self.get_video_info(video_path)
            if not video_info:
                return False, {}, {}
            
            # 2. 提取音频
            audio_success, audio_path, audio_data = self.extract_audio_from_video(video_path)
            
            # 3. 提取关键帧
            frames_success, frame_paths, frame_data = self.extract_frames_from_video(video_path)
            
            # 4. 处理视频（如果需要）
            video_tensor = None
            if hasattr(self.separation_config, 'video_processing') and self.separation_config.video_processing:
                print(f"\n🎬 处理视频张量...")
                try:
                    # 使用原有的视频处理逻辑
                    from qwen_omni_utils import process_mm_info
                    
                    # 创建视频处理参数
                    video_params = self._get_video_params()
                    
                    # 处理视频
                    video_conversation = self._add_video_to_conversation(conversation, video_path, video_params)
                    
                    # 使用qwen_omni_utils处理
                    audios, images, videos, video_kwargs = process_mm_info(
                        video_conversation, 
                        use_audio_in_video=False, 
                        return_video_kwargs=True
                    )
                    
                    if videos and len(videos) > 0:
                        video_tensor = videos[0]
                        print(f"  ✅ 视频张量处理成功: {video_tensor.shape}")
                    else:
                        print(f"  ⚠️ 视频张量处理失败")
                        
                except Exception as e:
                    print(f"  ❌ 视频张量处理失败: {e}")
            else:
                print(f"\n🎬 视频处理已禁用（配置中video_processing=False）")
            
            # 5. 收集处理结果
            processing_time = time.time() - start_time
            
            results = {
                'video_info': video_info,
                'audio_extraction': {
                    'success': audio_success,
                    'path': audio_path,
                    'data_shape': audio_data.shape if audio_data is not None else None
                },
                'frame_extraction': {
                    'success': frames_success,
                    'paths': frame_paths,
                    'count': len(frame_data),
                    'data_shapes': [frame.shape for frame in frame_data] if frame_data else []
                },
                'video_processing': {
                    'success': video_tensor is not None,
                    'tensor_shape': list(video_tensor.shape) if video_tensor is not None else None
                },
                'processing_time': processing_time,
                'total_time': time.time() - start_time
            }
            
            # 6. 打印处理总结
            self._print_processing_summary(results)
            
            return True, results, {'video_tensor': video_tensor, 'audio_data': audio_data, 'frame_data': frame_data}
            
        except Exception as e:
            print(f"❌ 分离处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False, {}, {}
    
    def _get_video_params(self) -> Dict[str, Any]:
        """获取视频处理参数"""
        params = {}
        
        # 帧数控制
        if self.video_config.nframes is not None:
            params['nframes'] = self.video_config.nframes
        elif self.video_config.fps is not None:
            params['fps'] = self.video_config.fps
            params['min_frames'] = self.video_config.min_frames
            params['max_frames'] = self.video_config.max_frames
        
        # 分辨率控制
        params['resized_height'] = self.video_config.resized_height
        params['resized_width'] = self.video_config.resized_width
        
        # 时间控制
        params['video_start'] = self.video_config.video_start
        params['video_end'] = self.video_config.video_end
        
        # 像素限制
        if self.video_config.max_pixels:
            params['max_pixels'] = self.video_config.max_pixels
        
        return params
    
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
    
    def _print_processing_summary(self, results: Dict[str, Any]):
        """打印处理总结"""
        print(f"\n📊 分离处理总结")
        print(f"{'='*60}")
        
        video_info = results['video_info']
        audio_extraction = results['audio_extraction']
        frame_extraction = results['frame_extraction']
        video_processing = results['video_processing']
        
        print(f"📹 视频信息:")
        print(f"  - 文件大小: {video_info['file_size_mb']:.2f} MB")
        print(f"  - 分辨率: {video_info['resolution']}")
        print(f"  - 时长: {video_info['duration']:.2f}秒")
        print(f"  - 音频: {'有' if video_info['has_audio'] else '无'}")
        
        print(f"\n🎵 音频提取:")
        print(f"  - 状态: {'✅ 成功' if audio_extraction['success'] else '❌ 失败'}")
        if audio_extraction['success']:
            print(f"  - 路径: {audio_extraction['path']}")
            print(f"  - 数据形状: {audio_extraction['data_shape']}")
        
        print(f"\n🖼️ 帧提取:")
        print(f"  - 状态: {'✅ 成功' if frame_extraction['success'] else '❌ 失败'}")
        if frame_extraction['success']:
            print(f"  - 提取帧数: {frame_extraction['count']}")
            print(f"  - 保存路径: {', '.join([Path(p).name for p in frame_extraction['paths']])}")
        
        print(f"\n🎬 视频处理:")
        print(f"  - 状态: {'✅ 成功' if video_processing['success'] else '❌ 失败'}")
        if video_processing['success']:
            print(f"  - 张量形状: {video_processing['tensor_shape']}")
        
        print(f"\n⏱️ 处理时间:")
        print(f"  - 分离处理: {results['processing_time']:.2f}秒")
        print(f"  - 总耗时: {results['total_time']:.2f}秒")
    
    def cleanup(self):
        """清理资源"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 清理中间文件
        if self.separation_config.cleanup_after_processing:
            print("🧹 清理中间文件...")
            # 这里可以添加清理逻辑

class EnhancedVideoOptimizationPresets:
    """增强版视频优化预设配置"""
    
    @staticmethod
    def get_separation_preset(name: str) -> Tuple[VideoOptimizationConfig, AudioVideoSeparationConfig]:
        """获取分离处理预设配置"""
        presets = {
            'audio_focus': (
                VideoOptimizationConfig(
                    nframes=2,
                    resized_height=112,
                    resized_width=112,
                    video_start=0.0,
                    video_end=2.0,
                    max_pixels=64 * 28 * 28,
                    use_half_precision=True,
                    enable_audio=False
                ),
                AudioVideoSeparationConfig(
                    extract_audio=True,
                    extract_frames=True,
                    frame_extraction_method="keyframes",
                    num_keyframes=2,
                    video_processing=False,  # 不处理视频，只提取音频和帧
                    save_intermediate=True
                )
            ),
            
            'frame_focus': (
                VideoOptimizationConfig(
                    nframes=4,
                    resized_height=168,
                    resized_width=168,
                    video_start=0.0,
                    video_end=3.0,
                    max_pixels=128 * 28 * 28,
                    use_half_precision=True,
                    enable_audio=False
                ),
                AudioVideoSeparationConfig(
                    extract_audio=True,
                    extract_frames=True,
                    frame_extraction_method="uniform",
                    num_keyframes=5,
                    video_processing=True,
                    save_intermediate=True
                )
            ),
            
            'balanced_separation': (
                VideoOptimizationConfig(
                    nframes=6,
                    resized_height=168,
                    resized_width=168,
                    video_start=0.0,
                    video_end=4.0,
                    max_pixels=256 * 28 * 28,
                    use_half_precision=True,
                    enable_audio=False
                ),
                AudioVideoSeparationConfig(
                    extract_audio=True,
                    extract_frames=True,
                    frame_extraction_method="keyframes",
                    num_keyframes=3,
                    video_processing=True,
                    save_intermediate=True
                )
            ),
            
            'full_extraction': (
                VideoOptimizationConfig(
                    nframes=8,
                    resized_height=224,
                    resized_width=224,
                    video_start=0.0,
                    video_end=5.0,
                    max_pixels=512 * 28 * 28,
                    use_half_precision=True,
                    enable_audio=False
                ),
                AudioVideoSeparationConfig(
                    extract_audio=True,
                    extract_frames=True,
                    frame_extraction_method="uniform",
                    num_keyframes=8,
                    video_processing=True,
                    save_intermediate=True
                )
            )
        }
        
        if name not in presets:
            raise ValueError(f"未知的预设配置: {name}")
        
        return presets[name]
    
    @staticmethod
    def list_separation_presets() -> List[str]:
        """列出所有可用的分离处理预设"""
        return ['audio_focus', 'frame_focus', 'balanced_separation', 'full_extraction']

def load_config_from_file(config_path: str) -> Tuple[VideoOptimizationConfig, AudioVideoSeparationConfig]:
    """从文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        video_config = VideoOptimizationConfig(**config_data.get('video', {}))
        separation_config = AudioVideoSeparationConfig(**config_data.get('separation', {}))
        
        return video_config, separation_config
        
    except Exception as e:
        print(f"配置加载失败: {e}")
        # 返回默认配置
        return VideoOptimizationConfig(), AudioVideoSeparationConfig()

def save_config_to_file(video_config: VideoOptimizationConfig, 
                        separation_config: AudioVideoSeparationConfig, 
                        config_path: str):
    """保存配置到文件"""
    try:
        config_data = {
            'video': {
                'nframes': video_config.nframes,
                'resized_height': video_config.resized_height,
                'resized_width': video_config.resized_width,
                'video_start': video_config.video_start,
                'video_end': video_config.video_end,
                'max_pixels': video_config.max_pixels,
                'use_half_precision': video_config.use_half_precision,
                'enable_audio': video_config.enable_audio
            },
            'separation': {
                'extract_audio': separation_config.extract_audio,
                'audio_sample_rate': separation_config.audio_sample_rate,
                'audio_format': separation_config.audio_format,
                'extract_frames': separation_config.extract_frames,
                'frame_extraction_method': separation_config.frame_extraction_method,
                'num_keyframes': separation_config.num_keyframes,
                'video_processing': separation_config.video_processing,
                'output_dir': separation_config.output_dir,
                'save_intermediate': separation_config.save_intermediate
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 配置已保存到: {config_path}")
        
    except Exception as e:
        print(f"❌ 配置保存失败: {e}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频输入优化器
专为显存受限环境设计，支持视频压缩、抽帧等优化策略
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
import tempfile
import logging
import json

# 添加qwen-omni-utils到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "qwen-omni-utils" / "src"))

try:
    from qwen_omni_utils.v2_5.vision_process import smart_resize, fetch_video
    QWEN_UTILS_AVAILABLE = True
except ImportError:
    QWEN_UTILS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("qwen_omni_utils不可用，将使用基础实现")

logger = logging.getLogger(__name__)

@dataclass
class VideoOptimizationConfig:
    """视频优化配置"""
    # 帧数控制
    max_frames: int = 4                    # 最大帧数
    target_fps: float = 2.0                # 目标帧率
    min_frames: int = 2                    # 最小帧数
    
    # 分辨率控制
    max_resolution: int = 224              # 最大分辨率（长边）
    min_resolution: int = 112              # 最小分辨率（短边）
    ensure_even_dimensions: bool = True    # 确保尺寸为偶数
    
    # 像素控制（用于qwen-omni-utils兼容）
    max_pixels: int = 224 * 224 * 4       # 最大像素数
    min_pixels: int = 112 * 112 * 2       # 最小像素数
    
    # 质量控制
    compression_quality: int = 85          # JPEG压缩质量
    video_bitrate: str = "500k"           # 视频比特率
    
    # 内存优化
    use_half_precision: bool = True        # 使用半精度
    batch_processing: bool = False         # 批处理模式
    clear_cache_between_frames: bool = True # 帧间清理缓存
    
    # 输出设置
    output_format: str = "mp4"            # 输出格式
    save_intermediate: bool = False        # 保存中间文件

class VideoOptimizer:
    """视频优化器"""
    
    def __init__(self, config: VideoOptimizationConfig):
        self.config = config
        self.temp_dir = Path(tempfile.mkdtemp())
        self.setup_environment()
    
    def setup_environment(self):
        """设置环境"""
        # 设置CUDA内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 创建临时目录
        (self.temp_dir / "frames").mkdir(exist_ok=True)
        (self.temp_dir / "videos").mkdir(exist_ok=True)
        
        logger.info(f"视频优化器初始化完成，临时目录: {self.temp_dir}")
    
    def get_optimal_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """计算最优分辨率"""
        # 使用qwen-omni-utils的智能调整（如果可用）
        if QWEN_UTILS_AVAILABLE:
            try:
                new_height, new_width = smart_resize(
                    height, width,
                    factor=28,  # qwen模型的factor
                    min_pixels=self.config.min_pixels,
                    max_pixels=self.config.max_pixels
                )
                return new_width, new_height
            except Exception as e:
                logger.warning(f"qwen智能调整失败，使用基础方法: {e}")
        
        # 基础调整方法
        max_dim = max(width, height)
        min_dim = min(width, height)
        
        if max_dim > self.config.max_resolution:
            scale = self.config.max_resolution / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
        elif min_dim < self.config.min_resolution:
            scale = self.config.min_resolution / min_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width, new_height = width, height
        
        # 确保尺寸为偶数
        if self.config.ensure_even_dimensions:
            new_width = new_width - new_width % 2
            new_height = new_height - new_height % 2
        
        return new_width, new_height
    
    def optimize_video_for_memory(self, video_path: str) -> str:
        """
        为内存优化视频
        
        Args:
            video_path: 原始视频路径
            
        Returns:
            优化后的视频路径
        """
        try:
            logger.info(f"开始优化视频: {video_path}")
            
            # 获取视频信息
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")
            
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / original_fps if original_fps > 0 else 0
            
            cap.release()
            
            # 计算优化参数
            target_width, target_height = self.get_optimal_resolution(width, height)
            
            # 计算目标帧数和帧率
            if total_frames > self.config.max_frames:
                target_frame_count = self.config.max_frames
                target_fps = target_frame_count / duration if duration > 0 else self.config.target_fps
            else:
                target_frame_count = total_frames
                target_fps = original_fps
            
            # 生成输出路径
            video_name = Path(video_path).stem
            output_path = self.temp_dir / "videos" / f"{video_name}_optimized.{self.config.output_format}"
            
            # 使用ffmpeg进行优化
            self._optimize_with_ffmpeg(
                video_path, str(output_path),
                target_width, target_height,
                target_fps, self.config.video_bitrate
            )
            
            # 验证输出
            if not output_path.exists():
                raise ValueError("视频优化失败")
            
            # 获取优化后的信息
            optimized_info = self._get_optimized_info(str(output_path))
            
            logger.info(f"视频优化完成:")
            logger.info(f"  原始: {width}x{height}, {total_frames}帧, {original_fps:.2f}fps")
            logger.info(f"  优化: {target_width}x{target_height}, {optimized_info['frames']}帧, {optimized_info['fps']:.2f}fps")
            logger.info(f"  文件大小: {os.path.getsize(video_path)/(1024*1024):.2f}MB -> {os.path.getsize(output_path)/(1024*1024):.2f}MB")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"视频优化失败: {e}")
            # 如果优化失败，返回原始视频
            logger.warning("使用原始视频")
            return video_path
    
    def _optimize_with_ffmpeg(self, input_path: str, output_path: str,
                            width: int, height: int, fps: float, bitrate: str):
        """使用ffmpeg优化视频"""
        try:
            import ffmpeg
            
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(
                stream,
                output_path,
                vf=f'scale={width}:{height}',
                r=fps,
                b_v=bitrate,
                crf=28,  # 恒定质量因子
                preset='ultrafast',  # 快速编码
                movflags='faststart',  # 优化流媒体
                loglevel='quiet'
            )
            
            ffmpeg.run(stream, overwrite_output=True)
            
        except ImportError:
            logger.warning("ffmpeg-python不可用，使用OpenCV处理")
            self._optimize_with_opencv(input_path, output_path, width, height, fps)
    
    def _optimize_with_opencv(self, input_path: str, output_path: str,
                            width: int, height: int, fps: float):
        """使用OpenCV优化视频（备选方案）"""
        try:
            # 读取原始视频
            cap = cv2.VideoCapture(input_path)
            
            # 设置输出视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            target_frame_interval = max(1, int(cap.get(cv2.CAP_PROP_FPS) / fps))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 跳帧处理
                if frame_count % target_frame_interval == 0:
                    # 调整帧大小
                    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    out.write(resized_frame)
                    
                    # 检查是否达到最大帧数
                    if frame_count // target_frame_interval >= self.config.max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            out.release()
            
        except Exception as e:
            logger.error(f"OpenCV视频优化失败: {e}")
            raise
    
    def _get_optimized_info(self, video_path: str) -> Dict[str, Any]:
        """获取优化后视频信息"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            return {
                'fps': fps,
                'frames': frames,
                'width': width,
                'height': height,
                'duration': frames / fps if fps > 0 else 0
            }
        except:
            return {}
    
    def create_qwen_compatible_input(self, video_path: str) -> Dict[str, Any]:
        """
        创建与qwen-omni-utils兼容的输入格式
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            qwen兼容的输入字典
        """
        try:
            if not QWEN_UTILS_AVAILABLE:
                logger.warning("qwen-omni-utils不可用，无法创建兼容输入")
                return {}
            
            # 优化视频
            optimized_video = self.optimize_video_for_memory(video_path)
            
            # 创建qwen格式的输入
            video_input = {
                "type": "video",
                "video": optimized_video,
                "nframes": self.config.max_frames,
                "resized_height": self.config.max_resolution,
                "resized_width": self.config.max_resolution,
                "max_pixels": self.config.max_pixels,
                "min_pixels": self.config.min_pixels
            }
            
            return video_input
            
        except Exception as e:
            logger.error(f"创建qwen兼容输入失败: {e}")
            return {}
    
    def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("临时文件清理完成")
        except Exception as e:
            logger.error(f"清理失败: {e}")

class MemoryOptimizedVideoHandler:
    """内存优化的视频处理器"""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.optimizer = VideoOptimizer(VideoOptimizationConfig())
    
    def estimate_video_memory_usage(self, video_path: str) -> Dict[str, float]:
        """估算视频内存使用量"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # 估算内存使用（RGB, float32）
            bytes_per_pixel = 3 * 4  # RGB * float32
            frame_memory_mb = (width * height * bytes_per_pixel) / (1024 * 1024)
            total_memory_mb = frame_memory_mb * min(total_frames, self.optimizer.config.max_frames)
            
            return {
                'frame_memory_mb': frame_memory_mb,
                'total_memory_mb': total_memory_mb,
                'recommended_frames': min(total_frames, int(self.max_memory_mb / frame_memory_mb)),
                'needs_optimization': total_memory_mb > self.max_memory_mb
            }
            
        except Exception as e:
            logger.error(f"内存估算失败: {e}")
            return {'needs_optimization': True}
    
    def auto_optimize_for_memory(self, video_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        根据内存限制自动优化视频
        
        Args:
            video_path: 原始视频路径
            
        Returns:
            (optimized_path, optimization_info): 优化后路径和优化信息
        """
        try:
            # 估算内存使用
            memory_info = self.estimate_video_memory_usage(video_path)
            
            if not memory_info.get('needs_optimization', False):
                logger.info("视频无需优化")
                return video_path, {'optimized': False, 'reason': '内存使用在限制内'}
            
            # 调整优化配置
            if memory_info.get('recommended_frames'):
                self.optimizer.config.max_frames = min(
                    self.optimizer.config.max_frames,
                    memory_info['recommended_frames']
                )
            
            # 进一步调整分辨率（如果内存仍然不够）
            if memory_info.get('total_memory_mb', 0) > self.max_memory_mb * 2:
                self.optimizer.config.max_resolution = min(
                    self.optimizer.config.max_resolution,
                    160  # 进一步降低分辨率
                )
            
            # 执行优化
            optimized_path = self.optimizer.optimize_video_for_memory(video_path)
            
            optimization_info = {
                'optimized': True,
                'original_memory_mb': memory_info.get('total_memory_mb', 0),
                'max_frames': self.optimizer.config.max_frames,
                'max_resolution': self.optimizer.config.max_resolution,
                'optimization_reason': '内存限制'
            }
            
            return optimized_path, optimization_info
            
        except Exception as e:
            logger.error(f"自动优化失败: {e}")
            return video_path, {'optimized': False, 'error': str(e)}
    
    def cleanup(self):
        """清理资源"""
        self.optimizer.cleanup()

# 预设配置
class VideoOptimizationPresets:
    """视频优化预设配置"""
    
    @staticmethod
    def get_preset(name: str) -> VideoOptimizationConfig:
        """获取预设配置"""
        presets = {
            'ultra_low_memory': VideoOptimizationConfig(
                max_frames=2,
                target_fps=1.0,
                max_resolution=112,
                max_pixels=112 * 112 * 2,
                compression_quality=75,
                video_bitrate="200k"
            ),
            
            'low_memory': VideoOptimizationConfig(
                max_frames=4,
                target_fps=2.0,
                max_resolution=168,
                max_pixels=168 * 168 * 4,
                compression_quality=80,
                video_bitrate="350k"
            ),
            
            'balanced': VideoOptimizationConfig(
                max_frames=6,
                target_fps=2.0,
                max_resolution=224,
                max_pixels=224 * 224 * 6,
                compression_quality=85,
                video_bitrate="500k"
            ),
            
            'high_quality': VideoOptimizationConfig(
                max_frames=8,
                target_fps=3.0,
                max_resolution=336,
                max_pixels=336 * 336 * 8,
                compression_quality=90,
                video_bitrate="800k"
            )
        }
        
        if name not in presets:
            raise ValueError(f"未知预设: {name}. 可用预设: {list(presets.keys())}")
        
        return presets[name]
    
    @staticmethod
    def list_presets() -> List[str]:
        """列出所有可用预设"""
        return ['ultra_low_memory', 'low_memory', 'balanced', 'high_quality']
    
    @staticmethod
    def get_preset_info() -> Dict[str, Dict[str, Any]]:
        """获取预设信息"""
        return {
            'ultra_low_memory': {
                'description': '超低内存模式',
                'memory_usage': '< 100MB',
                'quality': '低',
                'frames': 2,
                'resolution': '112x112'
            },
            'low_memory': {
                'description': '低内存模式', 
                'memory_usage': '< 200MB',
                'quality': '中低',
                'frames': 4,
                'resolution': '168x168'
            },
            'balanced': {
                'description': '平衡模式',
                'memory_usage': '< 400MB', 
                'quality': '中',
                'frames': 6,
                'resolution': '224x224'
            },
            'high_quality': {
                'description': '高质量模式',
                'memory_usage': '< 800MB',
                'quality': '高',
                'frames': 8,
                'resolution': '336x336'
            }
        }

def save_optimization_config(config: VideoOptimizationConfig, config_path: str):
    """保存优化配置到文件"""
    try:
        config_dict = {
            'max_frames': config.max_frames,
            'target_fps': config.target_fps,
            'min_frames': config.min_frames,
            'max_resolution': config.max_resolution,
            'min_resolution': config.min_resolution,
            'max_pixels': config.max_pixels,
            'min_pixels': config.min_pixels,
            'compression_quality': config.compression_quality,
            'video_bitrate': config.video_bitrate,
            'use_half_precision': config.use_half_precision,
            'output_format': config.output_format
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置已保存: {config_path}")
        
    except Exception as e:
        logger.error(f"配置保存失败: {e}")

def load_optimization_config(config_path: str) -> VideoOptimizationConfig:
    """从文件加载优化配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return VideoOptimizationConfig(**config_dict)
        
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return VideoOptimizationConfig()  # 返回默认配置
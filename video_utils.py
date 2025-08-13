#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理工具模块
支持视频音频分离、关键帧提取等功能
"""

import cv2
import numpy as np
import librosa
import soundfile as sf
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import os
import tempfile
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    """视频处理器，支持音频提取和帧提取"""
    
    def __init__(self, output_dir: str = "./extracted_media"):
        self.output_dir = Path(output_dir)
        self.setup_output_dirs()
    
    def setup_output_dirs(self):
        """创建输出目录"""
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "frames").mkdir(exist_ok=True)
    
    def extract_audio_from_video(self, video_path: str, sample_rate: int = 16000) -> Tuple[str, np.ndarray]:
        """
        从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            sample_rate: 目标采样率
            
        Returns:
            (audio_path, audio_data): 音频文件路径和音频数据
        """
        try:
            logger.info(f"从视频提取音频: {video_path}")
            
            # 使用librosa提取音频
            audio_data, sr = librosa.load(video_path, sr=sample_rate)
            
            # 生成输出文件名
            video_name = Path(video_path).stem
            audio_path = self.output_dir / "audio" / f"{video_name}_audio.wav"
            
            # 保存音频
            sf.write(str(audio_path), audio_data, sample_rate)
            
            logger.info(f"音频提取成功: {audio_path}, 时长: {len(audio_data)/sample_rate:.2f}秒")
            return str(audio_path), audio_data
            
        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            raise
    
    def extract_last_frame(self, video_path: str) -> Tuple[str, np.ndarray]:
        """
        提取视频的最后一帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            (frame_path, frame_data): 帧文件路径和帧数据
        """
        try:
            logger.info(f"提取最后一帧: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")
            
            # 获取总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError("视频没有有效帧")
            
            # 定位到最后一帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, frame = cap.read()
            
            if not ret:
                raise ValueError("无法读取最后一帧")
            
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 生成输出文件名
            video_name = Path(video_path).stem
            frame_path = self.output_dir / "frames" / f"{video_name}_last_frame.jpg"
            
            # 保存图像
            Image.fromarray(frame_rgb).save(frame_path, quality=95)
            
            cap.release()
            
            logger.info(f"最后一帧提取成功: {frame_path}")
            return str(frame_path), frame_rgb
            
        except Exception as e:
            logger.error(f"帧提取失败: {e}")
            raise
    
    def extract_uniform_frames(self, video_path: str, num_frames: int = 8) -> Tuple[List[str], List[np.ndarray]]:
        """
        均匀提取视频帧
        
        Args:
            video_path: 视频文件路径
            num_frames: 要提取的帧数
            
        Returns:
            (frame_paths, frame_data_list): 帧文件路径列表和帧数据列表
        """
        try:
            logger.info(f"均匀提取{num_frames}帧: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                raise ValueError("视频没有有效帧")
            
            # 计算帧索引
            frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
            
            frame_paths = []
            frame_data_list = []
            video_name = Path(video_path).stem
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # 转换BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 生成输出文件名
                    timestamp = frame_idx / fps if fps > 0 else 0
                    frame_path = self.output_dir / "frames" / f"{video_name}_frame_{i:02d}_{timestamp:.1f}s.jpg"
                    
                    # 保存图像
                    Image.fromarray(frame_rgb).save(frame_path, quality=95)
                    
                    frame_paths.append(str(frame_path))
                    frame_data_list.append(frame_rgb)
                    
                    logger.debug(f"提取帧 {i+1}/{len(frame_indices)}: {frame_path.name}")
            
            cap.release()
            
            logger.info(f"帧提取完成: {len(frame_paths)}/{len(frame_indices)} 帧成功")
            return frame_paths, frame_data_list
            
        except Exception as e:
            logger.error(f"帧提取失败: {e}")
            raise
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """获取视频基本信息"""
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
            has_audio = self._check_audio_track(video_path)
            
            cap.release()
            
            return {
                'file_path': video_path,
                'file_size_mb': os.path.getsize(video_path) / (1024 * 1024),
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration': duration,
                'has_audio': has_audio,
                'resolution': f"{width}x{height}",
                'aspect_ratio': width / height if height > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            return {}
    
    def _check_audio_track(self, video_path: str) -> bool:
        """检查视频是否有音频轨道"""
        try:
            audio, _ = librosa.load(video_path, sr=None, duration=1.0)
            return len(audio) > 0
        except:
            return False
    
    def process_video_for_model(self, video_path: str, 
                               extract_audio: bool = True,
                               extract_last_frame: bool = True,
                               max_frames: int = 8) -> Dict[str, Any]:
        """
        为模型输入处理视频
        
        Args:
            video_path: 视频文件路径
            extract_audio: 是否提取音频
            extract_last_frame: 是否提取最后一帧（True）或均匀提取帧（False）
            max_frames: 最大帧数（当extract_last_frame=False时使用）
            
        Returns:
            处理结果字典
        """
        results = {
            'video_info': self.get_video_info(video_path),
            'audio_path': None,
            'audio_data': None,
            'frame_paths': [],
            'frame_data': []
        }
        
        try:
            # 提取音频
            if extract_audio and results['video_info'].get('has_audio', False):
                audio_path, audio_data = self.extract_audio_from_video(video_path)
                results['audio_path'] = audio_path
                results['audio_data'] = audio_data
            
            # 提取帧
            if extract_last_frame:
                frame_path, frame_data = self.extract_last_frame(video_path)
                results['frame_paths'] = [frame_path]
                results['frame_data'] = [frame_data]
            else:
                frame_paths, frame_data_list = self.extract_uniform_frames(video_path, max_frames)
                results['frame_paths'] = frame_paths
                results['frame_data'] = frame_data_list
            
            results['success'] = True
            logger.info(f"视频处理完成: 音频={'有' if results['audio_path'] else '无'}, 帧数={len(results['frame_paths'])}")
            
        except Exception as e:
            logger.error(f"视频处理失败: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results

class OptimizedVideoProcessor:
    """优化的视频处理器，针对显存限制优化"""
    
    def __init__(self, max_pixels_per_frame: int = 112*112, max_frames: int = 4):
        self.max_pixels_per_frame = max_pixels_per_frame
        self.max_frames = max_frames
        
    def resize_frame_for_memory(self, frame: np.ndarray, max_pixels: int) -> np.ndarray:
        """调整帧大小以控制内存使用"""
        h, w = frame.shape[:2]
        current_pixels = h * w
        
        if current_pixels <= max_pixels:
            return frame
        
        # 计算缩放比例
        scale = np.sqrt(max_pixels / current_pixels)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 确保尺寸是偶数（某些编码器要求）
        new_h = new_h - new_h % 2
        new_w = new_w - new_w % 2
        
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"帧大小调整: {w}x{h} -> {new_w}x{new_h}")
        
        return resized_frame
    
    def extract_optimized_frames(self, video_path: str) -> List[np.ndarray]:
        """提取优化的视频帧"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 计算帧间隔
            if total_frames <= self.max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # 转换BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 调整大小以控制内存
                    optimized_frame = self.resize_frame_for_memory(frame_rgb, self.max_pixels_per_frame)
                    frames.append(optimized_frame)
            
            cap.release()
            logger.info(f"优化帧提取完成: {len(frames)}帧, 每帧最大像素: {self.max_pixels_per_frame}")
            
            return frames
            
        except Exception as e:
            logger.error(f"优化帧提取失败: {e}")
            raise

def create_conversation_with_video(video_path: str, 
                                 prompt: str = None,
                                 extract_audio: bool = True,
                                 extract_last_frame: bool = True) -> List[Dict]:
    """
    创建包含视频的对话格式
    
    Args:
        video_path: 视频文件路径
        prompt: 用户提示（可选）
        extract_audio: 是否提取音频
        extract_last_frame: 是否只提取最后一帧
        
    Returns:
        对话格式列表
    """
    try:
        # 处理视频
        processor = VideoProcessor()
        results = processor.process_video_for_model(
            video_path, 
            extract_audio=extract_audio,
            extract_last_frame=extract_last_frame
        )
        
        if not results['success']:
            raise ValueError(f"视频处理失败: {results.get('error', '未知错误')}")
        
        # 构建对话内容
        content = []
        
        # 添加视频内容
        if extract_audio and results['audio_path']:
            # 音频+图片模式
            content.append({
                "type": "audio",
                "audio": results['audio_path']
            })
            
            if results['frame_paths']:
                content.append({
                    "type": "image", 
                    "image": results['frame_paths'][0]
                })
        else:
            # 纯视频模式
            content.append({
                "type": "video",
                "video": video_path
            })
        
        # 添加文本提示
        if prompt:
            content.append({
                "type": "text",
                "text": prompt
            })
        
        # 构建对话
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
        
    except Exception as e:
        logger.error(f"创建对话失败: {e}")
        raise

def preprocess_video_for_low_vram(video_path: str, 
                                max_resolution: int = 224,
                                max_frames: int = 4,
                                compression_quality: int = 85) -> str:
    """
    为低显存环境预处理视频
    
    Args:
        video_path: 原始视频路径
        max_resolution: 最大分辨率（宽或高的最大值）
        max_frames: 最大帧数
        compression_quality: 压缩质量(1-100)
        
    Returns:
        处理后的视频路径
    """
    try:
        logger.info(f"为低显存预处理视频: {video_path}")
        
        # 创建临时输出文件
        output_path = tempfile.mktemp(suffix='.mp4')
        
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # 计算目标分辨率
        if max(width, height) > max_resolution:
            scale = max_resolution / max(width, height)
            target_width = int(width * scale)
            target_height = int(height * scale)
            # 确保尺寸是偶数
            target_width = target_width - target_width % 2
            target_height = target_height - target_height % 2
        else:
            target_width, target_height = width, height
        
        # 计算帧率（如果帧数超过限制）
        if total_frames > max_frames:
            target_fps = fps * max_frames / total_frames
        else:
            target_fps = fps
        
        # 使用ffmpeg压缩视频
        import ffmpeg
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(
            stream, 
            output_path,
            vf=f'scale={target_width}:{target_height}',
            r=target_fps,
            crf=28,  # 压缩率
            preset='ultrafast',
            movflags='faststart'
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        # 验证输出文件
        if not os.path.exists(output_path):
            raise ValueError("视频压缩失败")
        
        logger.info(f"视频预处理完成: {output_path}")
        logger.info(f"分辨率: {width}x{height} -> {target_width}x{target_height}")
        logger.info(f"帧率: {fps:.2f} -> {target_fps:.2f}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"视频预处理失败: {e}")
        # 如果处理失败，返回原始视频
        return video_path
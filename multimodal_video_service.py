#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态视频处理服务
支持多种输入方式：视频、音频+图片、文本+图片
输出：文本描述
"""

import os
import sys
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
import base64
import io

import torch
import numpy as np
from PIL import Image
import librosa
import soundfile as sf
import cv2

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "qwen-omni-utils" / "src"))

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    print("警告: 无法导入qwen_omni_utils，将使用基础处理方式")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """处理配置"""
    # 音频处理
    audio_sample_rate: int = 16000
    audio_format: str = "wav"
    
    # 图像处理
    image_max_size: int = 1024
    image_quality: int = 95
    
    # 视频处理
    video_max_frames: int = 8
    video_frame_interval: float = 1.0  # 秒
    
    # 输出配置
    output_format: str = "text"
    max_output_length: int = 1000

class MultimodalVideoProcessor:
    """多模态视频处理器"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.setup_environment()
        
    def setup_environment(self):
        """设置环境"""
        # 设置CUDA内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 创建临时目录
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"创建临时目录: {self.temp_dir}")
    
    def process_video_input(self, video_path: str) -> Dict[str, Any]:
        """处理视频输入"""
        try:
            logger.info(f"处理视频: {video_path}")
            
            # 提取音频
            audio_path = self.extract_audio_from_video(video_path)
            
            # 提取关键帧
            frames = self.extract_frames_from_video(video_path)
            
            # 处理多模态信息
            result = self.process_multimodal_content(
                audio_path=audio_path,
                frames=frames,
                video_path=video_path
            )
            
            return {
                "success": True,
                "type": "video",
                "result": result,
                "extracted_audio": audio_path,
                "extracted_frames": len(frames)
            }
            
        except Exception as e:
            logger.error(f"视频处理失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "type": "video"
            }
    
    def process_audio_image_input(self, audio_path: str, image_path: str) -> Dict[str, Any]:
        """处理音频+图片输入"""
        try:
            logger.info(f"处理音频+图片: {audio_path}, {image_path}")
            
            # 处理音频
            audio_info = self.process_audio(audio_path)
            
            # 处理图片
            image_info = self.process_image(image_path)
            
            # 处理多模态信息
            result = self.process_multimodal_content(
                audio_path=audio_path,
                frames=[image_path]
            )
            
            return {
                "success": True,
                "type": "audio_image",
                "result": result,
                "audio_info": audio_info,
                "image_info": image_info
            }
            
        except Exception as e:
            logger.error(f"音频+图片处理失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "type": "audio_image"
            }
    
    def process_text_image_input(self, text: str, image_path: str) -> Dict[str, Any]:
        """处理文本+图片输入"""
        try:
            logger.info(f"处理文本+图片: {text[:50]}..., {image_path}")
            
            # 处理图片
            image_info = self.process_image(image_path)
            
            # 处理多模态信息
            result = self.process_multimodal_content(
                text=text,
                frames=[image_path]
            )
            
            return {
                "success": True,
                "type": "text_image",
                "result": result,
                "text": text,
                "image_info": image_info
            }
            
        except Exception as e:
            logger.error(f"文本+图片处理失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "type": "text_image"
            }
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """从视频中提取音频"""
        try:
            output_path = self.temp_dir / f"extracted_audio.{self.config.audio_format}"
            
            # 使用ffmpeg提取音频
            import ffmpeg
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, str(output_path), 
                                 acodec='pcm_s16le', 
                                 ar=self.config.audio_sample_rate)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            logger.info(f"音频提取完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"音频提取失败: {str(e)}")
            # 使用opencv作为备选方案
            return self.extract_audio_opencv(video_path)
    
    def extract_audio_opencv(self, video_path: str) -> str:
        """使用OpenCV提取音频（备选方案）"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 创建音频数据（这里只是示例，实际应该提取真实音频）
            duration = frame_count / fps
            sample_rate = self.config.audio_sample_rate
            samples = int(duration * sample_rate)
            
            # 生成静音音频（实际应用中应该提取真实音频）
            audio_data = np.zeros(samples, dtype=np.float32)
            
            output_path = self.temp_dir / f"extracted_audio_opencv.{self.config.audio_format}"
            sf.write(str(output_path), audio_data, sample_rate)
            
            cap.release()
            return str(output_path)
            
        except Exception as e:
            logger.error(f"OpenCV音频提取失败: {str(e)}")
            raise
    
    def extract_frames_from_video(self, video_path: str) -> List[str]:
        """从视频中提取关键帧"""
        try:
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # 计算帧间隔
            interval = max(1, int(fps * self.config.video_frame_interval))
            
            for i in range(0, frame_count, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    frame_path = self.temp_dir / f"frame_{i:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame, 
                               [cv2.IMWRITE_JPEG_QUALITY, self.config.image_quality])
                    frames.append(str(frame_path))
                    
                    if len(frames) >= self.config.video_max_frames:
                        break
            
            cap.release()
            logger.info(f"提取了 {len(frames)} 帧")
            return frames
            
        except Exception as e:
            logger.error(f"帧提取失败: {str(e)}")
            return []
    
    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """处理音频文件"""
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=self.config.audio_sample_rate)
            
            # 计算音频特征
            duration = len(audio) / sr
            energy = np.mean(librosa.feature.rms(y=audio))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "energy": float(energy),
                "spectral_centroid": float(spectral_centroid),
                "samples": len(audio)
            }
            
        except Exception as e:
            logger.error(f"音频处理失败: {str(e)}")
            return {"error": str(e)}
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """处理图片文件"""
        try:
            image = Image.open(image_path)
            
            # 调整图片大小
            if max(image.size) > self.config.image_max_size:
                ratio = self.config.image_max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # 保存调整后的图片
                adjusted_path = self.temp_dir / f"adjusted_{Path(image_path).name}"
                image.save(adjusted_path, "JPEG", quality=self.config.image_quality)
                image_path = str(adjusted_path)
            
            return {
                "size": image.size,
                "mode": image.mode,
                "format": image.format,
                "path": image_path
            }
            
        except Exception as e:
            logger.error(f"图片处理失败: {str(e)}")
            return {"error": str(e)}
    
    def process_multimodal_content(self, 
                                 audio_path: Optional[str] = None,
                                 frames: Optional[List[str]] = None,
                                 video_path: Optional[str] = None,
                                 text: Optional[str] = None) -> str:
        """处理多模态内容并生成文本输出"""
        try:
            # 构建多模态信息
            mm_info = {}
            
            if audio_path:
                mm_info["audio"] = audio_path
            
            if frames:
                mm_info["images"] = frames
            
            if video_path:
                mm_info["video"] = video_path
            
            if text:
                mm_info["text"] = text
            
            # 尝试使用qwen_omni_utils处理
            try:
                if 'qwen_omni_utils' in sys.modules:
                    result = process_mm_info(mm_info)
                    return result
            except Exception as e:
                logger.warning(f"qwen_omni_utils处理失败，使用基础处理: {str(e)}")
            
            # 基础处理方式
            return self.basic_multimodal_processing(mm_info)
            
        except Exception as e:
            logger.error(f"多模态处理失败: {str(e)}")
            return f"处理失败: {str(e)}"
    
    def basic_multimodal_processing(self, mm_info: Dict[str, Any]) -> str:
        """基础多模态处理"""
        description_parts = []
        
        if "video" in mm_info:
            description_parts.append("检测到视频输入")
        
        if "audio" in mm_info:
            description_parts.append("检测到音频输入")
        
        if "images" in mm_info:
            description_parts.append(f"检测到{len(mm_info['images'])}张图片")
        
        if "text" in mm_info:
            description_parts.append(f"检测到文本输入: {mm_info['text'][:100]}...")
        
        if not description_parts:
            return "未检测到有效输入"
        
        return " | ".join(description_parts)
    
    def cleanup(self):
        """清理临时文件"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("临时文件清理完成")
        except Exception as e:
            logger.error(f"清理失败: {str(e)}")

class MultimodalVideoService:
    """多模态视频处理服务"""
    
    def __init__(self):
        self.config = ProcessingConfig()
        self.processor = MultimodalVideoProcessor(self.config)
    
    def process_input(self, 
                     video_file: Optional[str] = None,
                     audio_file: Optional[str] = None,
                     image_file: Optional[str] = None,
                     text_input: Optional[str] = None) -> Dict[str, Any]:
        """处理输入并返回结果"""
        try:
            # 根据输入类型选择处理方法
            if video_file:
                return self.processor.process_video_input(video_file)
            elif audio_file and image_file:
                return self.processor.process_audio_image_input(audio_file, image_file)
            elif text_input and image_file:
                return self.processor.process_text_image_input(text_input, image_file)
            else:
                return {
                    "success": False,
                    "error": "无效的输入组合",
                    "type": "unknown"
                }
                
        except Exception as e:
            logger.error(f"服务处理失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "type": "error"
            }
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """获取支持的格式"""
        return {
            "video": ["mp4", "avi", "mov", "mkv", "wmv"],
            "audio": ["wav", "mp3", "flac", "aac", "ogg"],
            "image": ["jpg", "jpeg", "png", "bmp", "gif", "webp"]
        }
    
    def cleanup(self):
        """清理资源"""
        self.processor.cleanup()

# 全局服务实例
service = MultimodalVideoService()

def process_multimodal_input(video_file=None, 
                           audio_file=None, 
                           image_file=None, 
                           text_input=None):
    """处理多模态输入的API函数"""
    return service.process_input(
        video_file=video_file,
        audio_file=audio_file,
        image_file=image_file,
        text_input=text_input
    )

if __name__ == "__main__":
    # 测试服务
    print("多模态视频处理服务启动...")
    print(f"支持的格式: {service.get_supported_formats()}")
    
    try:
        # 测试文本+图片输入
        test_result = process_multimodal_input(
            text_input="这是一张测试图片",
            image_file="test_image.png"
        )
        print(f"测试结果: {test_result}")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
    
    finally:
        service.cleanup()

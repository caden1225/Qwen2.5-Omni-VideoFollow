#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试音轨和视频分离处理功能
使用math.mp4文件进行测试，提取音频和最后一帧图像
"""

import torch
import gc
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig

# 从环境变量加载模型路径
MODEL_PATH = os.getenv('MODEL_PATH', "/home/caden/workplace/models/Qwen2.5-Omni-3B")

def print_gpu_memory_usage(stage=""):
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"[{stage}] GPU内存 - 已分配: {allocated:.2f}GB, 已预留: {reserved:.2f}GB")

def load_model():
    """加载模型"""
    print("=== 加载模型 ===")
    
    try:
        config = Qwen2_5OmniConfig.from_pretrained(MODEL_PATH)
        config.enable_audio_output = False
        
        device_map = {
            "thinker.model": "cuda",
            "thinker.lm_head": "cuda",
            "thinker.visual": "cuda",
            "thinker.audio_tower": "cuda",
        }
        
        max_memory = {0: "8GB", "cpu": "16GB"}
        
        print("正在加载模型...")
        print_gpu_memory_usage("加载前")
        
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            config=config,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
        
        print_gpu_memory_usage("模型加载后")
        print("✅ 模型和处理器加载成功")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def extract_audio_from_video(video_path: str, output_audio_path: str = None):
    """从视频中提取音频"""
    print(f"🎵 从视频中提取音频: {os.path.basename(video_path)}")
    
    if output_audio_path is None:
        output_audio_path = video_path.replace('.mp4', '_audio.wav')
    
    try:
        import cv2
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        # 获取音频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"  - 视频信息:")
        print(f"    FPS: {fps:.2f}")
        print(f"    ️总帧数: {total_frames}")
        print(f"    ⏱️时长: {duration:.2f}秒")
        
        # 检查是否有音频轨道
        has_audio = False
        try:
            # 尝试读取音频
            import librosa
            audio, sr = librosa.load(video_path, sr=None)
            has_audio = True
            print(f"  - 音频信息:")
            print(f"    🎵 采样率: {sr} Hz")
            print(f"    🔊 音频长度: {len(audio)/sr:.2f}秒")
            print(f"    📊 音频形状: {audio.shape}")
            
            # 保存音频（使用soundfile，因为librosa.output已被移除）
            import soundfile as sf
            sf.write(output_audio_path, audio, sr)
            print(f"  - ✅ 音频已保存到: {output_audio_path}")
            
        except Exception as e:
            print(f"  - ❌ 音频提取失败: {e}")
            has_audio = False
        
        cap.release()
        return has_audio, output_audio_path
        
    except Exception as e:
        print(f"❌ 视频处理失败: {e}")
        return False, None

def extract_last_frame_from_video(video_path: str, output_image_path: str = None):
    """从视频中提取最后一帧作为图像"""
    print(f"🖼️ 从视频中提取最后一帧: {os.path.basename(video_path)}")
    
    if output_image_path is None:
        output_image_path = video_path.replace('.mp4', '_last_frame.jpg')
    
    try:
        import cv2
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  - 视频信息:")
        print(f"    📐 分辨率: {width}x{height}")
        print(f"    🎬 总帧数: {total_frames}")
        print(f"    ⏱️ 时长: {total_frames/fps:.2f}秒")
        
        if total_frames > 0:
            # 跳转到最后一帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            
            # 读取最后一帧
            ret, frame = cap.read()
            
            if ret:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 保存图像
                cv2.imwrite(output_image_path, frame)
                print(f"  - ✅ 最后一帧已保存到: {output_image_path}")
                print(f"  - 📊 图像形状: {frame_rgb.shape}")
                
                cap.release()
                return True, output_image_path, frame_rgb
            else:
                print("  - ❌ 无法读取最后一帧")
                cap.release()
                return False, None, None
        else:
            print("  - ❌ 视频没有帧")
            cap.release()
            return False, None, None
            
    except Exception as e:
        print(f"❌ 帧提取失败: {e}")
        return False, None, None

def test_processor_capabilities(processor, video_path: str, audio_path: str, image_path: str):
    """测试processor处理音频、视频、图像的能力"""
    print(f"\n🧪 测试Processor能力")
    print(f"{'='*60}")
    
    # 测试1: 纯文本输入
    print(f"\n📝 测试1: 纯文本输入")
    try:
        text_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "你好，请介绍一下你自己。"}
                ],
            }
        ]
        
        text = processor.apply_chat_template(text_conversation, add_generation_prompt=True)
        print(f"  ✅ 文本处理成功，长度: {len(text)}")
        
    except Exception as e:
        print(f"  ❌ 文本处理失败: {e}")
    
    # 测试2: 图像输入
    print(f"\n🖼️ 测试2: 图像输入")
    try:
        if os.path.exists(image_path):
            from PIL import Image
            image = Image.open(image_path)
            print(f"  ✅ 图像加载成功: {image.size}")
            
            image_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "请描述这张图片。"}
                    ],
                }
            ]
            
            text = processor.apply_chat_template(image_conversation, add_generation_prompt=True)
            print(f"  ✅ 图像+文本处理成功，长度: {len(text)}")
            
        else:
            print(f"  ⚠️ 图像文件不存在: {image_path}")
            
    except Exception as e:
        print(f"  ❌ 图像处理失败: {e}")
    
    # 测试3: 音频输入
    print(f"\n🎵 测试3: 音频输入")
    try:
        if os.path.exists(audio_path):
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            print(f"  ✅ 音频加载成功: 采样率{sr}Hz, 长度{len(audio)/sr:.2f}秒")
            
            audio_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": "请描述这段音频。"}
                    ],
                }
            ]
            
            text = processor.apply_chat_template(audio_conversation, add_generation_prompt=True)
            print(f"  ✅ 音频+文本处理成功，长度: {len(text)}")
            
        else:
            print(f"  ⚠️ 音频文件不存在: {audio_path}")
            
    except Exception as e:
        print(f"  ❌ 音频处理失败: {e}")
    
    # 测试4: 视频输入
    print(f"\n🎬 测试4: 视频输入")
    try:
        if os.path.exists(video_path):
            video_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": "请描述这个视频。"}
                    ],
                }
            ]
            
            text = processor.apply_chat_template(video_conversation, add_generation_prompt=True)
            print(f"  ✅ 视频+文本处理成功，长度: {len(text)}")
            
        else:
            print(f"  ⚠️ 视频文件不存在: {video_path}")
            
    except Exception as e:
        print(f"  ❌ 视频处理失败: {e}")
    
    # 测试5: 混合输入（图像+音频+文本）
    print(f"\n🔀 测试5: 混合输入（图像+音频+文本）")
    try:
        if os.path.exists(image_path) and os.path.exists(audio_path):
            mixed_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": "请结合图像和音频内容进行描述。"}
                    ],
                }
            ]
            
            text = processor.apply_chat_template(mixed_conversation, add_generation_prompt=True)
            print(f"  ✅ 混合输入处理成功，长度: {len(text)}")
            
        else:
            print(f"  ⚠️ 图像或音频文件不存在，跳过混合输入测试")
            
    except Exception as e:
        print(f"  ❌ 混合输入处理失败: {e}")

def test_audio_video_separation():
    """测试音轨和视频分离处理"""
    print("🚀 音轨和视频分离处理测试")
    print("="*80)
    
    # 视频文件路径
    video_path = "/home/caden/workplace/qwen2.5-Omni_inference/test_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    print(f"📁 测试视频: {video_path}")
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"📊 文件大小: {file_size:.2f} MB")
    
    # 1. 提取音频
    print(f"\n{'='*60}")
    has_audio, audio_path = extract_audio_from_video(video_path)
    
    # 2. 提取最后一帧图像
    print(f"\n{'='*60}")
    frame_success, image_path, last_frame = extract_last_frame_from_video(video_path)
    
    # 3. 加载模型和处理器
    print(f"\n{'='*60}")
    model, processor = load_model()
    
    if model is None or processor is None:
        print("❌ 模型加载失败，无法继续测试")
        return
    
    # 4. 测试processor能力
    test_processor_capabilities(processor, video_path, audio_path, image_path)
    
    # 5. 测试分离处理的效果
    print(f"\n{'='*60}")
    print("🧪 测试分离处理效果")
    
    if has_audio and frame_success:
        print("✅ 成功提取音频和图像")
        
        # 测试音频+图像+文本的混合输入
        try:
            mixed_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": "这是一个数学教学视频的最后一帧图像和音频。请分析这个视频可能的内容。"}
                    ],
                }
            ]
            
            # 应用聊天模板
            text = processor.apply_chat_template(mixed_conversation, add_generation_prompt=True)
            print(f"✅ 混合输入模板应用成功")
            print(f"📝 模板长度: {len(text)} 字符")
            
            # 处理输入
            inputs = processor(
                text=[text],
                images=None,  # 这里我们分别处理图像和音频
                videos=None,  # 不使用视频输入
                padding=True,
                return_tensors="pt",
            )
            
            print(f"✅ 输入处理成功")
            print(f"📊 输入张量信息:")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape}, {value.dtype}")
                else:
                    print(f"  - {key}: {type(value)}")
            
        except Exception as e:
            print(f"❌ 分离处理测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 清理
    print(f"\n🧹 清理资源...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print_gpu_memory_usage("清理后")
    
    print(f"\n🎉 测试完成！")
    print(f"📋 测试总结:")
    print(f"  - 音频提取: {'✅ 成功' if has_audio else '❌ 失败'}")
    print(f"  - 图像提取: {'✅ 成功' if frame_success else '❌ 失败'}")
    print(f"  - 模型加载: {'✅ 成功' if model is not None else '❌ 失败'}")
    print(f"  - Processor测试: {'✅ 完成' if model is not None else '❌ 跳过'}")

if __name__ == "__main__":
    test_audio_video_separation()

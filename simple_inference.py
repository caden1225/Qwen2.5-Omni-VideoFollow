#!/usr/bin/env python3
"""
Qwen2.5-Omni 简化推理脚本
所有输入通过参数传递，不包含Web界面
"""

import io
import os
import argparse
import numpy as np
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

import time
import functools

class TimeitContext:
    def __init__(self, description):
        self.description = description
        self.start = None
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        print(f"{self.description} 执行耗时: {end - self.start:.6f} 秒")

def timeit(func_or_description):
    """支持装饰器和上下文管理器两种使用方式"""
    if callable(func_or_description):
        # 作为装饰器使用
        @functools.wraps(func_or_description)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func_or_description(*args, **kwargs)
            end = time.perf_counter()
            print(f"{func_or_description.__name__} 执行耗时: {end - start:.6f} 秒")
            return result
        return wrapper
    else:
        # 作为上下文管理器使用
        return TimeitContext(func_or_description)


class QwenOmniInference:
    def __init__(self, checkpoint_path, cpu_only=False, flash_attn2=False):
        """
        初始化模型和处理器
        
        Args:
            checkpoint_path: 模型检查点路径
            cpu_only: 是否只使用CPU
            flash_attn2: 是否启用flash attention 2
        """
        self.checkpoint_path = checkpoint_path
        self.cpu_only = cpu_only
        self.flash_attn2 = flash_attn2
        
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和处理器"""
        if self.cpu_only:
            device_map = 'cpu'
        else:
            device_map = 'auto'

        # 根据flash_attn2标志加载模型
        if self.flash_attn2:
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.checkpoint_path,
                torch_dtype='auto',
                attn_implementation='flash_attention_2',
                device_map=device_map
            )
        else:
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.checkpoint_path, 
                device_map=device_map, 
                torch_dtype='auto'
            )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.checkpoint_path)
        print(f"模型已加载到设备: {self.model.device}")
    
    def format_messages(self, messages, system_prompt):
        """
        格式化消息列表
        
        Args:
            messages: 消息列表
            system_prompt: 系统提示词
            
        Returns:
            格式化后的消息列表
        """
        formatted_messages = []
        formatted_messages.append({
            "role": "system", 
            "content": [{"type": "text", "text": system_prompt}]
        })
        
        for item in messages:
            if isinstance(item["content"], str):
                formatted_messages.append({
                    "role": item['role'], 
                    "content": item['content']
                })
            elif item["role"] == "user" and isinstance(item["content"], (list, tuple)):
                file_path = item["content"][0]
                
                # 根据文件类型添加相应的内容
                if os.path.exists(file_path):
                    if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        formatted_messages.append({
                            "role": item['role'],
                            "content": [{"type": "image", "image": file_path}]
                        })
                    elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        formatted_messages.append({
                            "role": item['role'],
                            "content": [{"type": "video", "video": file_path}]
                        })
                    elif file_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                        formatted_messages.append({
                            "role": item['role'],
                            "content": [{"type": "audio", "audio": file_path}]
                        })
        
        return formatted_messages
    
    def predict(self, messages, system_prompt, voice="Chelsie", save_audio_path=None):
        """
        执行推理
        
        Args:
            messages: 消息列表
            system_prompt: 系统提示词
            voice: 语音选择 ('Chelsie' 或 'Ethan')
            save_audio_path: 音频保存路径（可选）
            
        Returns:
            dict: 包含文本和音频的字典
        """
        # 格式化消息
        formatted_messages = self.format_messages(messages, system_prompt)
        
        # 应用聊天模板
        text = self.processor.apply_chat_template(
            formatted_messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # 处理多模态信息
        audios, images, videos = process_mm_info(
            formatted_messages, 
            use_audio_in_video=True
        )
        
        # 准备输入
        inputs = self.processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=True
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        
        with timeit("生成回复"):
            text_ids, audio = self.model.generate(
                **inputs, 
                speaker=voice, 
                use_audio_in_video=True
            )
        
        # 解码文本回复
        response = self.processor.batch_decode(
            text_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        response = response[0].split("\n")[-1]
        
        # 处理音频
        audio = np.array(audio * 32767).astype(np.int16)
        
        sf.write(save_audio_path, audio, samplerate=24000, format="WAV")
        print(f"音频已保存到: {save_audio_path}")
        
        return {
            "text": response,
            "audio": audio,
            "sample_rate": 24000
        }


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni 简化推理脚本")
    
    # 模型相关参数
    parser.add_argument('-c', '--checkpoint-path', type=str,
                       default='/home/caden/workplace/models/Qwen2.5-Omni-3B',
                       help='模型检查点路径')
    parser.add_argument('--cpu-only', action='store_true', 
                       help='仅使用CPU运行')
    parser.add_argument('--flash-attn2', action='store_true', 
                       help='启用flash attention 2')
    
    # 推理相关参数
    parser.add_argument('--system-prompt', type=str,
                       default='You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.',
                       help='系统提示词')
    parser.add_argument('--voice', type=str, choices=['Chelsie', 'Ethan'], 
                       default='Chelsie', help='语音选择')
    parser.add_argument('--save-audio', type=str, default="tmp.mp3",
                       help='音频保存路径（可选）')
    
    # 输入参数
    parser.add_argument('--text', type=str, help='文本输入')
    parser.add_argument('--audio', type=str, help='音频文件路径')
    parser.add_argument('--image', type=str, help='图像文件路径')
    parser.add_argument('--video', type=str, help='视频文件路径')
    
    args = parser.parse_args()
    args.audio = "/home/caden/workplace/Qwen2.5-Omni-VideoFollow/data/draw.mp4"

    # 初始化推理引擎
    print("正在加载模型...")
    inference = QwenOmniInference(
        checkpoint_path=args.checkpoint_path,
        cpu_only=args.cpu_only,
        flash_attn2=args.flash_attn2
    )
    
    # 构建消息列表
    messages = []
    
    if args.text:
        messages.append({"role": "user", "content": args.text})
    
    if args.audio:
        messages.append({"role": "user", "content": (args.audio,)})
    
    if args.image:
        messages.append({"role": "user", "content": (args.image,)})
    
    if args.video:
        messages.append({"role": "user", "content": (args.video,)})
    
    if not messages:
        print("错误：请提供至少一个输入（文本、音频、图像或视频）")
        return
    
    # 执行推理
    print("正在执行推理...")
    result = inference.predict(
        messages=messages,
        system_prompt=args.system_prompt,
        voice=args.voice,
        save_audio_path=args.save_audio
    )
    
    # 输出结果
    print("\n" + "="*50)
    print("推理结果:")
    print("="*50)
    print(f"文本回复: {result['text']}")
    print(f"音频采样率: {result['sample_rate']} Hz")
    print(f"音频长度: {len(result['audio'])} 采样点")
    if args.save_audio:
        print(f"音频已保存到: {args.save_audio}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Qwen2.5-Omni 多模态Gradio客户端界面
保留所有原有功能，但改为调用vLLM API服务
"""

import os
import time
import logging
import json
import requests
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import librosa
import cv2
from PIL import Image
import gradio as gr
from dotenv import load_dotenv
import soundfile as sf

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIClient:
    """API客户端类"""
    
    def __init__(self):
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.temp_dir = Path("temp_files")
        self.temp_dir.mkdir(exist_ok=True)
        
    def health_check(self) -> bool:
        """检查API服务健康状态"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200 and response.json().get("model_loaded", False)
        except Exception as e:
            print(f"❌ API健康检查失败: {e}")
            return False
    
    def process_multimodal(self, 
                          text_input: str,
                          image_input: Optional[Image.Image],
                          audio_input: Optional[str],
                          video_input: Optional[str],
                          system_prompt: str,
                          max_tokens: int,
                          extract_video_audio: bool,
                          extract_video_frame: bool,
                          using_mm_info_audio: bool,
                          enable_audio_output: bool = False) -> Tuple[str, str, Optional[str], Optional[str], float, float, Optional[str]]:
        """调用API处理多模态输入"""
        
        start_time = time.time()
        
        try:
            # 准备表单数据
            data = {
                'text_input': text_input if text_input and text_input.strip() else None,
                'system_prompt': system_prompt,
                'max_tokens': max_tokens,
                'extract_video_audio': extract_video_audio,
                'extract_video_frame': extract_video_frame,
                'using_mm_info_audio': using_mm_info_audio,
                'enable_audio_output': enable_audio_output
            }
            
            # 准备文件
            files = {}
            
            if image_input:
                # 保存图像到临时文件
                temp_image_path = self.temp_dir / f"temp_image_{int(time.time())}.png"
                image_input.save(temp_image_path)
                files['image_input'] = open(temp_image_path, 'rb')
            
            if audio_input:
                # 音频文件路径
                if os.path.exists(audio_input):
                    files['audio_input'] = open(audio_input, 'rb')
            
            if video_input:
                # 视频文件路径
                if os.path.exists(video_input):
                    files['video_input'] = open(video_input, 'rb')
            
            # 发送请求
            print(f"🚀 发送请求到API: {self.api_base_url}/process")
            response = requests.post(
                f"{self.api_base_url}/process",
                data=data,
                files=files,
                timeout=300  # 5分钟超时
            )
            
            # 关闭文件
            for file_obj in files.values():
                if hasattr(file_obj, 'close'):
                    file_obj.close()
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                
                # 处理提取的文件路径
                extracted_audio = None
                extracted_frame = None
                generated_audio = None
                
                if result.get('extracted_audio'):
                    # 下载提取的音频文件
                    extracted_audio = self._download_file(result['extracted_audio'], "audio")
                
                if result.get('extracted_frame'):
                    # 下载提取的图像文件
                    extracted_frame = self._download_file(result['extracted_frame'], "image")
                
                if result.get('generated_audio'):
                    # 下载生成的音频文件
                    generated_audio = self._download_file(result['generated_audio'], "audio")
                
                return (
                    result.get('status', '✅ 处理完成'),
                    result.get('response_text', ''),
                    extracted_audio,
                    extracted_frame,
                    processing_time,
                    result.get('peak_memory', 0),
                    generated_audio
                )
            else:
                error_msg = f"❌ API请求失败: {response.status_code} - {response.text}"
                print(error_msg)
                return error_msg, "", None, None, time.time() - start_time, 0, None
                
        except Exception as e:
            error_msg = f"❌ API调用失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, "", None, None, time.time() - start_time, 0, None
    
    def process_multimodal_streaming(self, 
                                   text_input: str,
                                   image_input: Optional[Image.Image],
                                   audio_input: Optional[str],
                                   video_input: Optional[str],
                                   system_prompt: str,
                                   max_tokens: int,
                                   extract_video_audio: bool,
                                   extract_video_frame: bool,
                                   using_mm_info_audio: bool,
                                   enable_audio_output: bool = False):
        """调用API进行流式处理"""
        
        try:
            # 准备表单数据
            data = {
                'text_input': text_input if text_input and text_input.strip() else None,
                'system_prompt': system_prompt,
                'max_tokens': max_tokens,
                'extract_video_audio': extract_video_audio,
                'extract_video_frame': extract_video_frame,
                'using_mm_info_audio': using_mm_info_audio,
                'enable_audio_output': enable_audio_output
            }
            
            # 准备文件
            files = {}
            
            if image_input:
                temp_image_path = self.temp_dir / f"temp_image_{int(time.time())}.png"
                image_input.save(temp_image_path)
                files['image_input'] = open(temp_image_path, 'rb')
            
            if audio_input:
                if os.path.exists(audio_input):
                    files['audio_input'] = open(audio_input, 'rb')
            
            if video_input:
                if os.path.exists(video_input):
                    files['video_input'] = open(video_input, 'rb')
            
            # 发送流式请求
            print(f"📡 发送流式请求到API: {self.api_base_url}/process_streaming")
            response = requests.post(
                f"{self.api_base_url}/process_streaming",
                data=data,
                files=files,
                stream=True,
                timeout=300
            )
            
            # 关闭文件
            for file_obj in files.values():
                if hasattr(file_obj, 'close'):
                    file_obj.close()
            
            if response.status_code == 200:
                # 处理流式响应
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            try:
                                data_str = line_str[6:]  # 去掉 'data: ' 前缀
                                result = json.loads(data_str)
                                
                                # 处理提取的文件
                                extracted_audio = None
                                extracted_frame = None
                                
                                if result.get('extracted_audio'):
                                    extracted_audio = self._download_file(result['extracted_audio'], "audio")
                                
                                if result.get('extracted_frame'):
                                    extracted_frame = self._download_file(result['extracted_frame'], "image")
                                
                                yield (
                                    result.get('status', '📡 流式处理中...'),
                                    result.get('response_text', ''),
                                    extracted_audio,
                                    extracted_frame,
                                    result.get('processing_time', 0),
                                    result.get('peak_memory', 0),
                                    None  # 流式模式下暂不支持音频输出
                                )
                            except json.JSONDecodeError as e:
                                print(f"⚠️ JSON解析失败: {e}")
                                continue
            else:
                error_msg = f"❌ 流式API请求失败: {response.status_code} - {response.text}"
                yield error_msg, "", None, None, 0, 0, None
                
        except Exception as e:
            error_msg = f"❌ 流式API调用失败: {str(e)}"
            print(error_msg)
            yield error_msg, "", None, None, 0, 0, None
    
    def _download_file(self, file_url: str, file_type: str) -> Optional[str]:
        """下载文件到本地临时目录"""
        try:
            # 从API获取文件
            response = requests.get(f"{self.api_base_url}/files/{file_type}/{file_url.split('/')[-1]}")
            if response.status_code == 200:
                # 保存到本地临时目录
                local_path = self.temp_dir / f"{file_type}_{int(time.time())}_{file_url.split('/')[-1]}"
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ 文件下载成功: {local_path}")
                return str(local_path)
            else:
                print(f"⚠️ 文件下载失败: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ 文件下载异常: {e}")
            return None

class MultimodalProcessor:
    """多模态处理器 - 客户端版本"""
    
    def __init__(self):
        self.api_client = APIClient()
        self.model_status = "⏳ 未连接"
        
    def check_api_connection(self) -> str:
        """检查API连接状态"""
        if self.api_client.health_check():
            self.model_status = "✅ 已连接到API服务"
            return self.model_status
        else:
            self.model_status = "❌ 无法连接到API服务"
            return self.model_status
    
    def get_model_status(self) -> str:
        """获取模型状态"""
        return self.model_status
    
    def process_multimodal(self, 
                          text_input: str,
                          image_input: Optional[Image.Image],
                          audio_input: Optional[str],
                          video_input: Optional[str],
                          system_prompt: str,
                          max_tokens: int,
                          extract_video_audio: bool,
                          extract_video_frame: bool,
                          using_mm_info_audio: bool,
                          enable_streaming: bool = False,
                          enable_audio_output: bool = False):
        """处理多模态输入 - 通过API"""
        
        if not self.api_client.health_check():
            return "❌ API服务未连接，请检查服务状态", "", None, None, 0, 0, None
        
        if enable_streaming:
            # 流式处理
            return self.api_client.process_multimodal_streaming(
                text_input, image_input, audio_input, video_input,
                system_prompt, max_tokens, extract_video_audio, extract_video_frame,
                using_mm_info_audio, enable_audio_output
            )
        else:
            # 标准处理
            return self.api_client.process_multimodal(
                text_input, image_input, audio_input, video_input,
                system_prompt, max_tokens, extract_video_audio, extract_video_frame,
                using_mm_info_audio, enable_audio_output
            )

# 创建处理器实例
processor = MultimodalProcessor()

# 构建Gradio界面
def create_interface():
    with gr.Blocks(title="Qwen2.5-Omni 多模态助手 (API客户端)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        ## 📡 通过vLLM API服务提供多模态AI能力
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔗 API连接控制")
                
                # API配置
                api_base_url = gr.Textbox(
                    label="🌐 API服务地址",
                    value=os.getenv("API_BASE_URL", "http://localhost:8000"),
                    placeholder="输入API服务地址，如: http://localhost:8000",
                    info="vLLM API服务的地址"
                )
                
                check_connection_btn = gr.Button("🔍 检查连接", variant="secondary")
                connection_status = gr.Textbox(
                    label="连接状态", 
                    value="⏳ 未连接", 
                    interactive=False
                )
                
                gr.Markdown("### ⚙️ 生成参数")
                system_prompt = gr.Textbox(
                    label="系统提示",
                    value="You are a helpful AI assistant.",
                    lines=2
                )
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=2048,
                    value=512,
                    step=50,
                    label="最大Token数"
                )
                
                gr.Markdown("### 🎬 视频处理选项")
                extract_video_audio = gr.Checkbox(
                    label="📢 提取视频音轨",
                    value=False,
                    info="将视频音轨提取为音频输入"
                )
                extract_video_frame = gr.Checkbox(
                    label="🖼️ 提取视频最后一帧",
                    value=False,
                    info="将视频最后一帧提取为图像输入"
                )
                using_mm_info_audio = gr.Checkbox(
                    label="🎵 使用mm_info提取音频",
                    value=False,
                    info="使用mm_info提取音频"
                )
                
                gr.Markdown("### ⚡ 输出模式")
                enable_streaming = gr.Checkbox(
                    label="📡 启用流式输出",
                    value=False,
                    info="实时逐步显示生成内容，提升交互体验"
                )
                enable_audio_output = gr.Checkbox(
                    label="🎵 启用语音输出",
                    value=False,
                    info="生成语音回答（如果API支持）"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📝 多模态输入")
                
                text_input = gr.Textbox(
                    label="💬 文本输入",
                    placeholder="输入您的问题或指令...",
                    lines=3
                )
                
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="🖼️ 图像输入",
                            type="pil"
                        )
                        
                        audio_input = gr.Audio(
                            label="🎵 音频输入",
                            type="filepath"
                        )
                    
                    with gr.Column():
                        video_input = gr.Video(
                            label="🎬 视频输入"
                        )

                with gr.Row():
                    process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ 清空", variant="secondary")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💬 生成结果")
                output_text = gr.Textbox(
                    label="AI回答",
                    lines=8,
                    placeholder="生成的回答将显示在这里...",
                    interactive=False
                )
            with gr.Column():
                # 显示生成的音频输出
                gr.Markdown("### 🎤 生成的语音回答")
                generated_audio_display = gr.Audio(
                    label="AI生成的语音回答",
                    visible=True,
                    interactive=False
                )
        # 显示提取的内容
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎵 提取的音频")
                extracted_audio_display = gr.Audio(
                    label="从视频提取的音频",
                    visible=True,
                    interactive=False
                )
            with gr.Column():
                gr.Markdown("### 🖼️ 提取的图像")
                extracted_image_display = gr.Image(
                    label="从视频提取的最后一帧",
                    type="pil",
                    visible=True,
                    interactive=False
                        )
            with gr.Column(scale=1):
                gr.Markdown("### 📊 处理信息")
                processing_info = gr.Textbox(
                    label="处理状态",
                    lines=8,
                    interactive=False,
                    value="等待处理..."
                )
        

        # 事件绑定
        def check_api_connection(api_url):
            """检查API连接"""
            processor.api_client.api_base_url = api_url
            return processor.check_api_connection()
        
        check_connection_btn.click(
            fn=check_api_connection,
            inputs=[api_base_url],
            outputs=[connection_status]
        )
        
        def handle_process_standard(text_input, image_input, audio_input, video_input, system_prompt, max_tokens, 
                                   extract_video_audio, extract_video_frame, using_mm_info_audio, enable_audio_output):
            """标准处理函数"""
            result = processor.process_multimodal(
                text_input, image_input, audio_input, video_input,
                system_prompt, max_tokens, extract_video_audio, extract_video_frame, using_mm_info_audio, False, enable_audio_output
            )
            return result[0], result[1], result[2], result[3], result[6]  # status, text, audio, image, generated_audio
        
        def handle_process_streaming(text_input, image_input, audio_input, video_input, system_prompt, max_tokens, 
                                   extract_video_audio, extract_video_frame, using_mm_info_audio, enable_audio_output):
            """流式处理函数"""
            for status, text, audio, image, time, memory, generated_audio in processor.process_multimodal_streaming(
                text_input, image_input, audio_input, video_input,
                system_prompt, max_tokens, extract_video_audio, extract_video_frame, using_mm_info_audio, enable_audio_output
            ):
                yield status, text, audio, image, generated_audio
        
        # 标准处理按钮
        process_btn.click(
            fn=handle_process_streaming if enable_streaming else handle_process_standard,
            inputs=[
                text_input, image_input, audio_input, video_input,
                system_prompt, max_tokens, extract_video_audio, extract_video_frame, using_mm_info_audio, enable_audio_output
            ],
            outputs=[processing_info, output_text, extracted_audio_display, extracted_image_display, generated_audio_display]
        )
        
        def clear_all():
            return "", None, None, None, "", "等待处理...", None, None, False, None
        
        clear_btn.click(
            fn=clear_all,
            outputs=[text_input, image_input, audio_input, video_input, output_text, processing_info, extracted_audio_display, extracted_image_display, enable_streaming, generated_audio_display]
        )
        
        # 页面加载时自动检查连接
        demo.load(
            fn=processor.check_api_connection,
            outputs=[connection_status]
        )
    
    return demo

if __name__ == "__main__":
    # 创建界面
    demo = create_interface()
    
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        debug=True,
        show_error=True
    )

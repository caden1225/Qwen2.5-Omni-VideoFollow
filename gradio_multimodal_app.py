#!/usr/bin/env python3
"""
Qwen2.5-Omni 多模态Gradio界面
支持视频、语音、图像、文本等不同模态的组合输入
"""

import os
import time
import logging
from typing import List, Optional, Tuple, Dict, Any
import tempfile

import torch
import numpy as np
import librosa
import cv2
from PIL import Image
import gradio as gr
from dotenv import load_dotenv
import soundfile as sf

# 导入qwen-omni-utils  
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor


try:
    from transformers import Qwen2_5OmniForConditionalGeneration
except ImportError:
    print("❌ 无法导入模型，请检查环境配置")
    raise

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalProcessor:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.getenv("MODEL_PATH", "/home/caden/models/Qwen2.5-Omni-3B")
        self.temp_files = []
        
    def load_model(self):
        """加载模型和处理器"""
        try:
            print(f"正在加载模型: {self.model_path}")
            # 标准模式但尝试优化显存
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            print("📦 模型加载完成")

            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
            print("处理器加载完成")
            return "✅ 模型加载成功"
            
        except Exception as e:
            error_msg = f"❌ 模型加载失败: {e}"
            print(error_msg)
            return error_msg

    def extract_video_features(self, video_path: str, extract_audio: bool = False, extract_frame: bool = False):
        """从视频中提取音频和最后一帧"""
        features = {}
        
        if extract_audio:
            try:
                audio, sr = librosa.load(video_path, sr=16000)
                features['audio'] = audio
                print(f"提取音频成功: {len(audio)} samples at {sr}Hz")
            except Exception as e:
                print(f"音频提取失败: {e}")
        
        if extract_frame:
            try:
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                ret, frame = cap.read()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    features['last_frame'] = image
                    print("提取视频最后一帧成功")
                else:
                    print("提取帧失败")
                cap.release()
            except Exception as e:
                print(f"帧提取失败: {e}")
        
        return features

    def process_multimodal(self, 
                          text_input: str,
                          image_input: Optional[Image.Image],
                          audio_input: Optional[str],
                          video_input: Optional[str],
                          system_prompt: str,
                          max_tokens: int,
                          extract_video_audio: bool,
                          extract_video_frame: bool,
                          enable_streaming: bool = False):
        """处理多模态输入"""
        
        if self.model is None:
            return "❌ 请先加载模型", "", None, None, 0, 0
        
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        extracted_audio = None
        extracted_frame = None
        
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            ]
            
            user_content = []
            
            # 添加文本
            if text_input and text_input.strip():
                user_content.append({"type": "text", "text": text_input.strip()})
            
            # 处理视频
            if video_input:
                if extract_video_audio or extract_video_frame:
                    print(f"🎬 开始提取视频特征...")
                    features = self.extract_video_features(
                        video_input, 
                        extract_audio=extract_video_audio, 
                        extract_frame=extract_video_frame
                    )
                    
                    if 'audio' in features:
                        user_content.append({"type": "audio", "audio": features['audio']})
                        # 保存提取的音频供显示
                        temp_audio_path = f"temp_extracted_audio_{int(time.time())}.wav"
                        sf.write(temp_audio_path, features['audio'], 16000)
                        extracted_audio = temp_audio_path
                        print(f"✅ 音频已提取并保存: {temp_audio_path}")
                    
                    if 'last_frame' in features:
                        user_content.append({"type": "image", "image": features['last_frame']})
                        # 保存提取的图像供显示
                        extracted_frame = features['last_frame']
                        print(f"✅ 图像已提取")
                else:
                    user_content.append({"type": "video", "video": video_input})
            
            # 处理图像
            if image_input:
                user_content.append({"type": "image", "image": image_input})
            
            # 处理音频
            if audio_input:
                try:
                    audio_data, _ = librosa.load(audio_input, sr=16000)
                    user_content.append({"type": "audio", "audio": audio_data})
                except Exception as e:
                    print(f"音频处理失败: {e}")
            
            # 如果没有任何内容，使用默认
            if not user_content:
                user_content.append({"type": "text", "text": "Hello"})
            
            messages.append({"role": "user", "content": user_content})
            
            print(f"📝 构建的消息包含 {len(user_content)} 个内容项")
            
            # 应用聊天模板
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"📄 生成的prompt长度: {len(text_prompt)} 字符")
            
            # 处理多模态信息
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            print(f"📊 多模态处理结果: audios={len(audios) if audios else 0}, images={len(images) if images else 0}, videos={len(videos) if videos else 0}")
            
            # 处理输入
            inputs = self.processor(
                text=text_prompt, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True
            )
            
            print(f"🔧 输入tensor形状: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs.items()]}")
            
            # 标准模式：获取模型设备
            device = next(self.model.parameters()).device
            inputs = inputs.to(device).to(self.model.dtype)
            
            print("🚀 开始生成回答...")
            
            if enable_streaming:
                # 流式生成
                print("📡 使用流式输出...")
                response_text = ""
                
                # 获取输入长度用于后续过滤
                input_length = inputs['input_ids'].shape[1]
                
                with torch.no_grad():
                    # 使用流式生成
                    from transformers import TextIteratorStreamer
                    import threading
                    
                    streamer = TextIteratorStreamer(
                        self.processor.tokenizer, 
                        skip_prompt=True,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    
                    generation_kwargs = dict(
                        inputs=inputs,
                        streamer=streamer,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        use_audio_in_video=True,
                        return_audio=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
                    # 在单独线程中生成
                    thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    # 流式读取生成的文本
                    for new_text in streamer:
                        response_text += new_text
                        # 实时更新可以在这里处理，但gradio需要特殊处理
                    
                    thread.join()
                
                print(f"📡 流式生成完成，总长度: {len(response_text)} 字符")
                
            else:
                # 标准生成
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs, 
                        max_new_tokens=max_tokens,
                        do_sample=False,  # 使用贪心解码
                        use_audio_in_video=True,
                        return_audio=False,  # 禁用音频输出
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                print(f"📤 生成输出形状: {output.shape}")
                
                # 解码响应
                response_text = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                print(f"📝 原始回答长度: {len(response_text)} 字符")
                
                # 提取实际的回答部分 (去掉输入的prompt)
                # 找到assistant回答的开始位置
                if "<|im_start|>assistant" in response_text:
                    response_text = response_text.split("<|im_start|>assistant")[-1].strip()
                elif "assistant\n" in response_text:
                    response_text = response_text.split("assistant\n")[-1].strip()
                
                # 清理结束符号
                if response_text.endswith("<|im_end|>"):
                    response_text = response_text[:-10].strip()
            
            print(f"✅ 清理后回答: {response_text[:100]}...")
            
            # 计算统计信息
            processing_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            
            # 构建输出信息
            info_text = f"""
📊 **处理统计**
- ⏱️ 处理时间: {processing_time:.2f}秒
- 💾 峰值显存: {peak_memory:.1f}MB
- 🔤 最大Token数: {max_tokens}
- 📝 输出Token数: {len(output[0]) - len(inputs['input_ids'][0])}
- 🎯 系统提示: {system_prompt[:50]}...

📋 **输入内容**
- 文本输入: {'✅' if text_input else '❌'}
- 图像输入: {'✅' if image_input else '❌'}  
- 音频输入: {'✅' if audio_input else '❌'}
- 视频输入: {'✅' if video_input else '❌'}
- 提取音轨: {'✅' if extract_video_audio and extracted_audio else '❌'}
- 提取帧: {'✅' if extract_video_frame and extracted_frame else '❌'}
            """
            
            # 构建详细的处理信息
            status_info = f"""✅ 处理完成 - {'流式' if enable_streaming else '标准'}模式
⏱️ 处理时间: {processing_time:.2f}秒
💾 峰值显存: {peak_memory:.1f}MB"""

            return status_info, response_text, extracted_audio, extracted_frame, processing_time, peak_memory
            
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, "", None, None, 0, 0

    def process_multimodal_streaming(self, 
                                   text_input: str,
                                   image_input: Optional[Image.Image],
                                   audio_input: Optional[str],
                                   video_input: Optional[str],
                                   system_prompt: str,
                                   max_tokens: int,
                                   extract_video_audio: bool,
                                   extract_video_frame: bool):
        """流式处理多模态输入 - 使用生成器返回逐步更新"""
        
        if self.model is None:
            yield "❌ 请先加载模型", "", None, None
            return
        
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        extracted_audio = None
        extracted_frame = None
        
        try:
            # 前期处理 - 和普通处理相同
            yield "🔄 开始处理...", "", None, None
            
            # 构建消息
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            ]
            
            user_content = []
            
            # 添加文本
            if text_input and text_input.strip():
                user_content.append({"type": "text", "text": text_input.strip()})
            
            # 处理视频
            if video_input:
                if extract_video_audio or extract_video_frame:
                    yield "🎬 提取视频特征...", "", None, None
                    
                    features = self.extract_video_features(
                        video_input, 
                        extract_audio=extract_video_audio, 
                        extract_frame=extract_video_frame
                    )
                    
                    if 'audio' in features:
                        user_content.append({"type": "audio", "audio": features['audio']})
                        temp_audio_path = f"temp_extracted_audio_{int(time.time())}.wav"
                        sf.write(temp_audio_path, features['audio'], 16000)
                        extracted_audio = temp_audio_path
                    
                    if 'last_frame' in features:
                        user_content.append({"type": "image", "image": features['last_frame']})
                        extracted_frame = features['last_frame']
                        
                    yield "✅ 视频特征提取完成", "", extracted_audio, extracted_frame
                else:
                    user_content.append({"type": "video", "video": video_input})
            
            # 处理图像
            if image_input:
                user_content.append({"type": "image", "image": image_input})
            
            # 处理音频
            if audio_input:
                try:
                    audio_data, _ = librosa.load(audio_input, sr=16000)
                    user_content.append({"type": "audio", "audio": audio_data})
                except Exception as e:
                    print(f"音频处理失败: {e}")
            
            if not user_content:
                user_content.append({"type": "text", "text": "Hello"})
            
            messages.append({"role": "user", "content": user_content})
            
            yield "📝 构建多模态输入...", "", extracted_audio, extracted_frame
            
            # 应用聊天模板
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 处理多模态信息
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            
            # 处理输入
            inputs = self.processor(
                text=text_prompt, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True
            )
            
            device = next(self.model.parameters()).device
            inputs = inputs.to(device).to(self.model.dtype)
            
            yield "🚀 开始流式生成...", "", extracted_audio, extracted_frame
            
            # 流式生成
            from transformers import TextIteratorStreamer
            import threading
            
            streamer = TextIteratorStreamer(
                self.processor.tokenizer, 
                skip_prompt=True,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_audio_in_video=True,
                return_audio=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            
            # 在单独线程中生成
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 流式输出
            response_text = ""
            for new_text in streamer:
                if new_text.strip():  # 忽略空白token
                    response_text += new_text
                    processing_time = time.time() - start_time
                    status = f"📡 流式生成中... ({processing_time:.1f}s)"
                    yield status, response_text, extracted_audio, extracted_frame
            
            thread.join()
            
            # 最终结果
            processing_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            
            final_status = f"""✅ 流式生成完成!
⏱️ 总时间: {processing_time:.2f}秒
💾 峰值显存: {peak_memory:.1f}MB
📝 输出长度: {len(response_text)} 字符"""
            
            yield final_status, response_text, extracted_audio, extracted_frame
            
        except Exception as e:
            error_msg = f"❌ 流式处理失败: {str(e)}"
            yield error_msg, "", extracted_audio, extracted_frame


# 创建处理器实例
processor = MultimodalProcessor()

# 构建Gradio界面
def create_interface():
    with gr.Blocks(title="Qwen2.5-Omni 多模态助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Qwen2.5-Omni 多模态智能助手
        
        支持文本、图像、音频、视频的多种组合输入，智能生成回答。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ 模型控制")
                load_btn = gr.Button("🔄 加载模型", variant="primary")
                model_status = gr.Textbox(
                    label="模型状态", 
                    value="⏳ 未加载", 
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
                
                gr.Markdown("### ⚡ 输出模式")
                enable_streaming = gr.Checkbox(
                    label="📡 启用流式输出",
                    value=False,
                    info="实时逐步显示生成内容，提升交互体验"
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
                    stream_btn = gr.Button("📡 流式处理", variant="secondary", size="lg", visible=False)
                clear_btn = gr.Button("🗑️ 清空", variant="secondary")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 💬 生成结果")
                output_text = gr.Textbox(
                    label="AI回答",
                    lines=8,
                    placeholder="生成的回答将显示在这里...",
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
        load_btn.click(
            fn=processor.load_model,
            outputs=model_status
        )
        
        def handle_process_standard(text_input, image_input, audio_input, video_input, system_prompt, max_tokens, 
                                   extract_video_audio, extract_video_frame, enable_streaming):
            """标准处理函数"""
            if enable_streaming:
                # 如果启用流式，给出提示
                return "📡 流式模式：请点击下面的流式处理按钮", "", None, None
            else:
                # 使用标准处理
                result = processor.process_multimodal(
                    text_input, image_input, audio_input, video_input,
                    system_prompt, max_tokens, extract_video_audio, extract_video_frame, False
                )
                return result[0], result[1], result[2], result[3]  # status, text, audio, image
        
        def handle_process_streaming(text_input, image_input, audio_input, video_input, system_prompt, max_tokens, 
                                   extract_video_audio, extract_video_frame):
            """流式处理函数"""
            for status, text, audio, image in processor.process_multimodal_streaming(
                text_input, image_input, audio_input, video_input,
                system_prompt, max_tokens, extract_video_audio, extract_video_frame
            ):
                yield status, text, audio, image
        
        # 标准处理按钮
        process_btn.click(
            fn=handle_process_standard,
            inputs=[
                text_input, image_input, audio_input, video_input,
                system_prompt, max_tokens, extract_video_audio, extract_video_frame, enable_streaming
            ],
            outputs=[processing_info, output_text, extracted_audio_display, extracted_image_display]
        )
        
        # 流式按钮已在上面定义
        
        stream_btn.click(
            fn=handle_process_streaming,
            inputs=[
                text_input, image_input, audio_input, video_input,
                system_prompt, max_tokens, extract_video_audio, extract_video_frame
            ],
            outputs=[processing_info, output_text, extracted_audio_display, extracted_image_display]
        )
        
        # 根据流式开关控制按钮显示
        def update_buttons(enable_streaming):
            if enable_streaming:
                return gr.update(value="🚀 标准处理"), gr.update(visible=True)
            else:
                return gr.update(value="🚀 开始处理"), gr.update(visible=False)
        
        enable_streaming.change(
            fn=update_buttons,
            inputs=[enable_streaming],
            outputs=[process_btn, stream_btn]
        )
        
        def clear_all():
            return "", None, None, None, "", "等待处理...", None, None, False
        
        clear_btn.click(
            fn=clear_all,
            outputs=[text_input, image_input, audio_input, video_input, output_text, processing_info, extracted_audio_display, extracted_image_display, enable_streaming]
        )
    
    return demo


if __name__ == "__main__":
    # 创建界面
    demo = create_interface()
    
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True,
        show_error=True
    )
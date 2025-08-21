#!/usr/bin/env python3
"""
Qwen2.5-Omni 多模态Gradio界面
支持视频、语音、图像、文本等不同模态的组合输入
"""

import os
import time
import logging
from typing import List, Optional, Tuple, Dict, Any

import torch
from PIL import Image
import gradio as gr
from dotenv import load_dotenv
import soundfile as sf
import numpy as np
import functools

# 导入必要的模块
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VOICE_LIST = ['Chelsie', 'Ethan']
DEFAULT_VOICE = 'Chelsie'

# 时间统计装饰器和上下文管理器
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

        if self.flash_attn2:
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(self.checkpoint_path,
                                                        torch_dtype='auto',
                                                        attn_implementation='flash_attention_2',
                                                        device_map=device_map)
        else:
             self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(self.checkpoint_path, device_map=device_map, torch_dtype='auto')

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.checkpoint_path)
        print(f"模型已加载到设备: {self.model.device}")
    
    
    def predict(self, formatted_messages, voice="Chelsie", save_audio_path=None):
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
        
        print("🚀 开始生成回答...")
        with timeit("生成回复"):
            text_ids, audio_output = self.model.generate(
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
        if audio_output is not None and save_audio_path:
            audio = np.array(audio_output * 32767).astype(np.int16)
            sf.write(save_audio_path, audio, samplerate=24000, format="WAV")
            print(f"音频已保存到: {save_audio_path}")
        
        return {
            "text": response,
            "audio": audio,
            "sample_rate": 24000
        }

class MultimodalProcessor:
    def __init__(self):
        self.inference_engine = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.getenv("MODEL_PATH", "/home/caden/models/Qwen2.5-Omni-3B")
        self.temp_files = []
        
    def _print_memory_usage(self, stage: str):
        """打印不同阶段的显存占用统计"""
        if not torch.cuda.is_available():
            return
            
        print(f"\n🔍 {stage} - 显存占用统计:")
        print(f"   当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   显存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"   显存峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        # 获取GPU信息
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            free = total_memory - cached
            
            print(f"   GPU {i} ({props.name}):")
            print(f"     总显存: {total_memory:.2f} GB")
            print(f"     已分配: {allocated:.2f} GB")
            print(f"     已缓存: {cached:.2f} GB")
            print(f"     可用显存: {free:.2f} GB")
        print()
        
    def _analyze_module_memory_usage(self, stage: str):
        """分析Qwen2.5-Omni模型各个模块的显存/内存占用"""
        if self.inference_engine is None or self.inference_engine.model is None:
            return
            
        print(f"\n🔍 {stage} - 模块级资源占用分析:")
        
        # 定义主要模块及其路径
        main_modules = {
            "thinker": "thinker",
            "talker": "talker", 
            "token2wav": "token2wav",
            "thinker.model": "thinker.model",
            "thinker.visual": "thinker.visual",
            "thinker.audio_tower": "thinker.audio_tower",
            "thinker.lm_head": "thinker.lm_head"
        }
        
        total_params = 0
        total_trainable_params = 0
        
        for module_name, module_path in main_modules.items():
            try:
                # 获取模块对象
                module = self.inference_engine.model
                for attr in module_path.split('.'):
                    if hasattr(module, attr):
                        module = getattr(module, attr)
                    else:
                        module = None
                        break
                
                if module is not None:
                    # 统计参数数量
                    module_params = sum(p.numel() for p in module.parameters())
                    module_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    
                    # 统计显存占用
                    module_gpu_memory = 0
                    module_cpu_memory = 0
                    
                    for param in module.parameters():
                        if param.device.type == 'cuda':
                            module_gpu_memory += param.numel() * param.element_size()
                        else:
                            module_cpu_memory += param.numel() * param.element_size()
                    
                    # 统计缓冲区显存
                    for buffer in module.buffers():
                        if buffer.device.type == 'cuda':
                            module_gpu_memory += buffer.numel() * buffer.element_size()
                        else:
                            module_cpu_memory += buffer.numel() * buffer.element_size()
                    
                    total_params += module_params
                    total_trainable_params += module_trainable_params
                    
                    print(f"   📊 {module_name}:")
                    print(f"      参数数量: {module_params:,} ({module_params/1e6:.2f}M)")
                    print(f"      可训练参数: {module_trainable_params:,} ({module_trainable_params/1e6:.2f}M)")
                    print(f"      GPU显存: {module_gpu_memory/1024**3:.3f} GB")
                    print(f"      CPU内存: {module_cpu_memory/1024**3:.3f} GB")
                    
                    # 显示设备位置
                    if hasattr(module, 'device'):
                        print(f"      设备位置: {module.device}")
                    else:
                        # 尝试从参数推断设备
                        device = next(module.parameters()).device if list(module.parameters()) else "unknown"
                        print(f"      设备位置: {device}")
                        
                else:
                    print(f"   ❌ {module_name}: 模块不存在")
                    
            except Exception as e:
                print(f"   ❌ {module_name}: 分析失败 - {e}")
        
        print(f"\n   📈 总计:")
        print(f"      总参数: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"      可训练参数: {total_trainable_params:,} ({total_trainable_params/1e6:.2f}M)")
        
        # 显示模型总显存占用
        if torch.cuda.is_available():
            model_gpu_memory = sum(p.numel() * p.element_size() for p in self.inference_engine.model.parameters() if p.device.type == 'cuda')
            model_gpu_memory += sum(b.numel() * b.element_size() for b in self.inference_engine.model.buffers() if b.device.type == 'cuda')
            print(f"      模型总GPU显存: {model_gpu_memory/1024**3:.3f} GB")
        
        print()
        
    def _monitor_inference_memory(self, modality_name: str):
        """监控推理过程中特定模态的显存占用"""
        class MemoryMonitor:
            def __init__(self, name, processor):
                self.name = name
                self.processor = processor
                self.start_memory = 0
                self.peak_memory = 0
                self.activation_memory = 0
                self.hooks = []
                
            def __enter__(self):
                if torch.cuda.is_available():
                    # 记录开始时的显存状态
                    torch.cuda.empty_cache()
                    self.start_memory = torch.cuda.memory_allocated()
                    self.peak_memory = self.start_memory
                    
                    # 设置钩子来监控激活值
                    self._setup_hooks()
                    
                    print(f"🔍 开始监控 {self.name} 模态推理显存...")
                    print(f"   初始显存: {self.start_memory / 1024**3:.3f} GB")
                
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if torch.cuda.is_available():
                    # 移除钩子
                    self._remove_hooks()
                    
                    # 计算最终统计
                    final_memory = torch.cuda.memory_allocated()
                    total_peak = torch.cuda.max_memory_allocated()
                    
                    # 计算激活显存（推理过程中的临时显存）
                    self.activation_memory = max(0, total_peak - self.start_memory)
                    
                    print(f"🔍 {self.name} 模态推理完成 - 显存统计:")
                    print(f"   初始显存: {self.start_memory / 1024**3:.3f} GB")
                    print(f"   最终显存: {final_memory / 1024**3:.3f} GB")
                    print(f"   峰值显存: {total_peak / 1024**3:.3f} GB")
                    print(f"   激活显存: {self.activation_memory / 1024**3:.3f} GB")
                    print(f"   推理增量: {(final_memory - self.start_memory) / 1024**3:.3f} GB")
                    print()
                    
                    # 重置峰值统计
                    torch.cuda.reset_peak_memory_stats()
                
            def _setup_hooks(self):
                """设置钩子来监控激活值"""
                def forward_hook(module, input, output):
                    if torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated()
                        self.peak_memory = max(self.peak_memory, current_memory)
                        
                        # 计算激活显存
                        if hasattr(output, 'numel'):
                            if hasattr(output, 'element_size'):
                                activation_size = output.numel() * output.element_size()
                            else:
                                activation_size = output.numel() * 4  # 假设float32
                            
                            if output.device.type == 'cuda':
                                self.activation_memory = max(self.activation_memory, activation_size)
                
                # 为关键模块添加钩子
                if hasattr(self.processor.inference_engine.model, 'thinker'):
                    if hasattr(self.processor.inference_engine.model.thinker, 'visual'):
                        self.hooks.append(self.processor.inference_engine.model.thinker.visual.register_forward_hook(forward_hook))
                    if hasattr(self.processor.inference_engine.model.thinker, 'audio_tower'):
                        self.hooks.append(self.processor.inference_engine.model.thinker.audio_tower.register_forward_hook(forward_hook))
                    if hasattr(self.processor.inference_engine.model.thinker, 'model'):
                        self.hooks.append(self.processor.inference_engine.model.thinker.model.register_forward_hook(forward_hook))
                
                if hasattr(self.processor.inference_engine.model, 'talker'):
                    self.hooks.append(self.processor.inference_engine.model.talker.register_forward_hook(forward_hook))
                
                if hasattr(self.processor.inference_engine.model, 'token2wav'):
                    self.hooks.append(self.processor.inference_engine.model.token2wav.register_forward_hook(forward_hook))
            
            def _remove_hooks(self):
                """移除所有钩子"""
                for hook in self.hooks:
                    hook.remove()
                self.hooks.clear()
        
        return MemoryMonitor(modality_name, self)
        
    def load_model(self, checkpoint_path=None, cpu_only=False, flash_attn2=False):
        """加载模型和处理器"""
        try:
            if checkpoint_path is None:
                checkpoint_path = self.model_path
                
            print(f"🚀 正在加载模型: {checkpoint_path}")
            print(f"   设备: {'CPU' if cpu_only else 'GPU'}")
            print(f"   Flash Attention 2: {'启用' if flash_attn2 else '禁用'}")
            
            # 使用重写的推理引擎
            self.inference_engine = QwenOmniInference(
                checkpoint_path=checkpoint_path,
                cpu_only=cpu_only,
                flash_attn2=flash_attn2
            )
            
            print("📦 模型加载完成")

            # 统计显存占用
            if torch.cuda.is_available():
                self._print_memory_usage("模型加载完成后")
            
            # 分析各模块资源占用
            self._analyze_module_memory_usage("模型加载完成后")
            
            return "✅ 模型加载成功"
            
        except Exception as e:
            error_msg = f"❌ 模型加载失败: {e}"
            print(error_msg)
            return error_msg

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
        """处理多模态输入"""
        
        if self.inference_engine is None:
            return "❌ 请先加载模型", "", None, None, 0, 0, None
        
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        extracted_audio = None
        extracted_frame = None
        generated_audio = None
        
        modality_types = []
        if text_input and text_input.strip():
            modality_types.append("文本")
        if image_input:
            modality_types.append("图像")
        if audio_input:
            modality_types.append("音频")
        if video_input:
            modality_types.append("视频")
        
        modality_name = "+".join(modality_types) if modality_types else "纯文本"

        # 格式化消息

        formatted_messages = []
        formatted_messages.append({
            "role": "system", 
            "content": [{"type": "text", "text": system_prompt}]
        })
        
        if text_input:
            formatted_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": text_input}]
            })
        if image_input and image_input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            formatted_messages.append({
                "role": "user",
                "content": [{"type": "image", "image": image_input}]
            })
        if audio_input and audio_input.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
            formatted_messages.append({
                "role": "user",
                "content": [{"type": "audio", "audio": audio_input}]
            })
        if video_input and video_input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            formatted_messages.append({
                "role": "user",
                "content": [{"type": "video", "video": video_input}]
            })      
        
        # 生成临时音频文件路径
        temp_audio_path = f"temp_generated_audio_{int(time.time())}.wav" if enable_audio_output else None
        
        with self._monitor_inference_memory(f"{modality_name}-推理"):
            result = self.inference_engine.predict(
                formatted_messages=formatted_messages,
                voice=DEFAULT_VOICE,
                save_audio_path=temp_audio_path
            )
        
        response_text = result["text"]
        generated_audio = temp_audio_path if enable_audio_output and result.get("audio") is not None else None
        
        # 计算统计信息
        processing_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            
        # 构建详细的处理信息
        status_info = f"""✅ 处理完成
⏱️ 处理时间: {processing_time:.2f}秒
💾 峰值显存: {peak_memory:.1f}MB"""

        return status_info, response_text, extracted_audio, extracted_frame, processing_time, peak_memory, generated_audio
            

# 创建处理器实例
processor = MultimodalProcessor()

# 构建Gradio界面
def create_interface():
    with gr.Blocks(title="Qwen2.5-Omni 多模态助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Qwen2.5-Omni 多模态智能助手
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ 模型控制")
                
                # 模型路径输入
                model_path = gr.Textbox(
                    label="🔗 模型路径",
                    value="/home/caden/models/Qwen2.5-Omni-3B",
                    placeholder="输入模型检查点路径",
                    info="Qwen2.5-Omni模型的本地路径"
                )
                
                # 模型配置选项
                cpu_only = gr.Checkbox(
                    label="🖥️ 仅使用CPU",
                    value=False,
                    info="强制使用CPU运行模型"
                )
                
                flash_attn2 = gr.Checkbox(
                    label="⚡ Flash Attention 2",
                    value=False,
                    info="启用Flash Attention 2优化"
                )
                
                load_btn = gr.Button("🔄 加载模型", variant="primary")
                model_status = gr.Textbox(
                    label="模型状态", 
                    value="⏳ 未加载", 
                    interactive=False
                )
                
                gr.Markdown("### ⚙️ 生成参数")
                system_prompt = gr.Textbox(
                    label="系统提示",
                    value="'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'",
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
                    info="生成语音回答（如果模型支持）"
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
                
                # 显示生成的音频输出
                gr.Markdown("### 🎤 生成的语音回答")
                generated_audio_display = gr.Audio(
                    label="AI生成的语音回答",
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
        
        # 加载模型函数
        def load_model_with_config(model_path, cpu_only, flash_attn2):
            return processor.load_model(
                checkpoint_path=model_path,
                cpu_only=cpu_only,
                flash_attn2=flash_attn2
            )
        
        load_btn.click(
            fn=load_model_with_config,
            inputs=[model_path, cpu_only, flash_attn2],
            outputs=model_status
        )
        
        def handle_process(text_input, image_input, audio_input, video_input, system_prompt, max_tokens, 
                          extract_video_audio, extract_video_frame, using_mm_info_audio, enable_streaming, enable_audio_output):
            """处理多模态输入"""
            result = processor.process_multimodal(
                text_input, image_input, audio_input, video_input,
                system_prompt, max_tokens, extract_video_audio, extract_video_frame, using_mm_info_audio, enable_streaming, enable_audio_output
            )
            return result[0], result[1], result[2], result[3], result[6]  # status, text, audio, image, generated_audio
        
        # 处理按钮
        process_btn.click(
            fn=handle_process,
            inputs=[
                text_input, image_input, audio_input, video_input,
                system_prompt, max_tokens, extract_video_audio, extract_video_frame, using_mm_info_audio, enable_streaming, enable_audio_output
            ],
            outputs=[processing_info, output_text, extracted_audio_display, extracted_image_display, generated_audio_display]
        )
        
        def clear_all():
            return "", None, None, None, "", "等待处理...", None, None, None
        
        clear_btn.click(
            fn=clear_all,
            outputs=[text_input, image_input, audio_input, video_input, output_text, processing_info, extracted_audio_display, extracted_image_display, generated_audio_display]
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
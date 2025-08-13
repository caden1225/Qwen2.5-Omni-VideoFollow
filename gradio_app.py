#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-Omni 多模态推理交互界面
支持视频、音频+图片、文本+图片等多种输入方式
"""

import gradio as gr
import os
import sys
import json
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import threading
import time

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入自定义模块
from model_inference import ModelManager, InferencePresets
from memory_manager import MemoryPresets, ConfigurableMemoryLoader
from video_utils import VideoProcessor
from video_optimizer import VideoOptimizationPresets, MemoryOptimizedVideoHandler

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioApp:
    """Gradio应用主类"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.video_processor = VideoProcessor()
        self.memory_loader = ConfigurableMemoryLoader()
        self.current_model_loaded = False
        self.temp_files = []
        self.processing_lock = threading.Lock()
        
        # 加载默认模型
        self.initialize_model()
    
    def initialize_model(self):
        """初始化模型"""
        try:
            logger.info("初始化模型...")
            model_path = os.getenv("MODEL_PATH", "/home/caden/workplace/models/Qwen2.5-Omni-3B")
            
            if self.model_manager.load_model_with_config(
                *InferencePresets.get_model_preset("low_vram")
            ):
                self.current_model_loaded = True
                logger.info("模型初始化成功")
            else:
                logger.error("模型初始化失败")
                
        except Exception as e:
            logger.error(f"模型初始化异常: {e}")
    
    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(
            title="Qwen2.5-Omni 多模态推理",
            theme=gr.themes.Soft(),
            css=self.get_custom_css()
        ) as interface:
            
            # 标题
            gr.Markdown("""
            # 🤖 Qwen2.5-Omni 多模态推理平台
            
            支持多种输入方式的智能AI交互，包括视频分析、语音理解、图像识别等功能。
            """)
            
            # 模型配置区域
            with gr.Accordion("⚙️ 模型配置", open=False):
                model_configs = self.create_model_config_section()
            
            # 输入方式选择
            with gr.Row():
                input_mode = gr.Radio(
                    choices=["🎬 视频输入", "🎵 音频+图片", "📝 文本+图片", "💬 纯文本"],
                    value="🎬 视频输入",
                    label="输入方式",
                    info="选择您要使用的输入方式"
                )
            
            # 动态输入区域
            input_components = self.create_input_sections(input_mode)
            
            # 输出区域
            output_components = self.create_output_section()
            
            # 系统信息
            with gr.Accordion("📊 系统信息", open=False):
                system_components = self.create_system_info_section()
            
            # 设置事件处理器
            all_components = {
                **input_components,
                'outputs': output_components,
                'model_configs': model_configs,
                'system': system_components
            }
            self.setup_event_handlers(all_components)
        
        return interface
    
    def create_model_config_section(self):
        """创建模型配置区域"""
        with gr.Row():
            model_preset = gr.Dropdown(
                choices=InferencePresets.list_presets(),
                value="low_vram",
                label="模型预设",
                info="选择适合您硬件的预设配置"
            )
            
            memory_preset = gr.Dropdown(
                choices=MemoryPresets.list_presets(),
                value="low_vram", 
                label="内存预设",
                info="选择内存管理策略"
            )
        
        with gr.Row():
            load_model_btn = gr.Button("🔄 重新加载模型", variant="secondary")
            model_status = gr.Textbox(
                value="🟢 模型已就绪" if self.current_model_loaded else "🔴 模型未加载",
                label="模型状态",
                interactive=False
            )
        
        # 模型配置事件
        def reload_model(model_preset_val, memory_preset_val):
            try:
                with self.processing_lock:
                    # 清理当前模型
                    self.model_manager.cleanup()
                    
                    # 加载新配置
                    model_config, vram_config = InferencePresets.get_model_preset(model_preset_val)
                    memory_config = MemoryPresets.get_preset(memory_preset_val)
                    
                    # 更新内存加载器配置
                    self.memory_loader.config = memory_config
                    
                    # 重新加载模型
                    if self.model_manager.load_model_with_config(model_config, vram_config):
                        self.current_model_loaded = True
                        return "🟢 模型重新加载成功"
                    else:
                        self.current_model_loaded = False
                        return "🔴 模型加载失败"
                        
            except Exception as e:
                logger.error(f"模型重新加载失败: {e}")
                self.current_model_loaded = False
                return f"🔴 加载失败: {str(e)}"
        
        load_model_btn.click(
            fn=reload_model,
            inputs=[model_preset, memory_preset],
            outputs=[model_status]
        )
        
        return {
            'model_preset': model_preset,
            'memory_preset': memory_preset, 
            'model_status': model_status,
            'load_model_btn': load_model_btn
        }
    
    def create_input_sections(self, input_mode):
        """创建输入区域"""
        # 视频输入
        with gr.Group(visible=True) as video_group:
            gr.Markdown("### 🎬 视频输入")
            with gr.Row():
                with gr.Column(scale=2):
                    video_input = gr.Video(
                        label="上传视频文件",
                        info="支持MP4, AVI, MOV等格式"
                    )
                    
                    video_prompt = gr.Textbox(
                        label="提示词（可选）",
                        placeholder="请描述您希望AI关注的内容...",
                        lines=2
                    )
                
                with gr.Column(scale=1):
                    video_options = self.create_video_options()
            
            process_video_btn = gr.Button("🚀 分析视频", variant="primary", size="lg")
        
        # 音频+图片输入
        with gr.Group(visible=False) as audio_image_group:
            gr.Markdown("### 🎵 音频+图片输入")
            with gr.Row():
                audio_input = gr.Audio(
                    label="上传音频文件",
                    type="filepath"
                )
                image_input_1 = gr.Image(
                    label="上传图片",
                    type="filepath"
                )
            
            audio_image_prompt = gr.Textbox(
                label="提示词（可选）",
                placeholder="请描述您希望AI分析的内容...",
                lines=2
            )
            
            process_audio_image_btn = gr.Button("🚀 分析音频+图片", variant="primary", size="lg")
        
        # 文本+图片输入
        with gr.Group(visible=False) as text_image_group:
            gr.Markdown("### 📝 文本+图片输入")
            with gr.Row():
                text_input = gr.Textbox(
                    label="输入文本",
                    placeholder="请输入您要分析的文本内容...",
                    lines=4
                )
                image_input_2 = gr.Image(
                    label="上传图片",
                    type="filepath"
                )
            
            process_text_image_btn = gr.Button("🚀 分析文本+图片", variant="primary", size="lg")
        
        # 纯文本输入
        with gr.Group(visible=False) as text_group:
            gr.Markdown("### 💬 纯文本对话")
            text_only_input = gr.Textbox(
                label="输入文本",
                placeholder="请输入您的问题...",
                lines=3
            )
            
            process_text_btn = gr.Button("🚀 发送消息", variant="primary", size="lg")
        
        # 输入方式切换事件
        def switch_input_mode(mode):
            visibility = {
                "🎬 视频输入": (True, False, False, False),
                "🎵 音频+图片": (False, True, False, False), 
                "📝 文本+图片": (False, False, True, False),
                "💬 纯文本": (False, False, False, True)
            }
            
            return [gr.Group(visible=v) for v in visibility.get(mode, (True, False, False, False))]
        
        input_mode.change(
            fn=switch_input_mode,
            inputs=[input_mode],
            outputs=[video_group, audio_image_group, text_image_group, text_group]
        )
        
        return {
            'video_input': video_input,
            'video_prompt': video_prompt,
            'video_options': video_options,
            'process_video_btn': process_video_btn,
            'audio_input': audio_input,
            'image_input_1': image_input_1,
            'audio_image_prompt': audio_image_prompt,
            'process_audio_image_btn': process_audio_image_btn,
            'text_input': text_input,
            'image_input_2': image_input_2,
            'process_text_image_btn': process_text_image_btn,
            'text_only_input': text_only_input,
            'process_text_btn': process_text_btn
        }
    
    def create_video_options(self):
        """创建视频处理选项"""
        with gr.Group():
            gr.Markdown("#### 视频处理选项")
            
            with gr.Row():
                extract_audio = gr.Checkbox(
                    label="提取音频",
                    value=True,
                    info="从视频中提取音频进行分析"
                )
                
                extract_last_frame = gr.Checkbox(
                    label="仅最后一帧",
                    value=True,
                    info="只提取最后一帧图像（否则均匀提取多帧）"
                )
            
            with gr.Row():
                video_optimization = gr.Dropdown(
                    choices=VideoOptimizationPresets.list_presets(),
                    value="balanced",
                    label="视频优化预设",
                    info="选择视频处理优化级别"
                )
        
        return {
            'extract_audio': extract_audio,
            'extract_last_frame': extract_last_frame,
            'video_optimization': video_optimization
        }
    
    def create_output_section(self):
        """创建输出区域"""
        with gr.Group():
            gr.Markdown("### 📊 分析结果")
            
            with gr.Row():
                output_text = gr.Textbox(
                    label="AI分析结果",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    placeholder="分析结果将在这里显示..."
                )
            
            with gr.Tabs():
                with gr.Tab("详细信息"):
                    output_json = gr.JSON(label="详细结果")
                
                with gr.Tab("处理日志"):
                    processing_log = gr.Textbox(
                        label="处理日志",
                        lines=6,
                        interactive=False
                    )
                
                with gr.Tab("提取的媒体"):
                    with gr.Row():
                        extracted_audio = gr.Audio(label="提取的音频", visible=False)
                        extracted_images = gr.Gallery(
                            label="提取的图片",
                            visible=False,
                            columns=3,
                            rows=2
                        )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ 清除结果", variant="secondary")
                download_btn = gr.Button("💾 下载结果", variant="secondary")
        
        return {
            'output_text': output_text,
            'output_json': output_json,
            'processing_log': processing_log,
            'extracted_audio': extracted_audio,
            'extracted_images': extracted_images,
            'clear_btn': clear_btn,
            'download_btn': download_btn
        }
    
    def create_system_info_section(self):
        """创建系统信息区域"""
        with gr.Row():
            with gr.Column():
                memory_info = gr.JSON(
                    label="内存使用情况",
                    value=self.get_current_memory_info()
                )
                
                refresh_memory_btn = gr.Button("🔄 刷新内存信息", variant="secondary")
            
            with gr.Column():
                model_info = gr.JSON(
                    label="模型信息",
                    value=self.get_model_info()
                )
        
        # 刷新内存信息事件
        def refresh_memory():
            return self.get_current_memory_info()
        
        refresh_memory_btn.click(
            fn=refresh_memory,
            outputs=[memory_info]
        )
        
        return {
            'memory_info': memory_info,
            'model_info': model_info,
            'refresh_memory_btn': refresh_memory_btn
        }
    
    def get_custom_css(self) -> str:
        """获取自定义CSS"""
        return """
        .gradio-container {
            max-width: 1400px !important;
            margin: 0 auto !important;
        }
        
        .input-group {
            border: 2px solid #e1e5e9;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        }
        
        .output-group {
            border: 2px solid #28a745;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            background: linear-gradient(135deg, #f8fff9 0%, #ffffff 100%);
        }
        
        .config-group {
            border: 2px solid #ffc107;
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
            background: linear-gradient(135deg, #fffbf0 0%, #ffffff 100%);
        }
        
        .processing {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        """
    
    def setup_event_handlers(self, components: Dict[str, Any]):
        """设置事件处理器"""
        # 视频处理
        components['process_video_btn'].click(
            fn=self.process_video_input,
            inputs=[
                components['video_input'],
                components['video_prompt'],
                components['video_options']['extract_audio'],
                components['video_options']['extract_last_frame'],
                components['video_options']['video_optimization']
            ],
            outputs=[
                components['outputs']['output_text'],
                components['outputs']['output_json'],
                components['outputs']['processing_log'],
                components['outputs']['extracted_audio'],
                components['outputs']['extracted_images']
            ]
        )
        
        # 音频+图片处理
        components['process_audio_image_btn'].click(
            fn=self.process_audio_image_input,
            inputs=[
                components['audio_input'],
                components['image_input_1'],
                components['audio_image_prompt']
            ],
            outputs=[
                components['outputs']['output_text'],
                components['outputs']['output_json'],
                components['outputs']['processing_log']
            ]
        )
        
        # 文本+图片处理
        components['process_text_image_btn'].click(
            fn=self.process_text_image_input,
            inputs=[
                components['text_input'],
                components['image_input_2']
            ],
            outputs=[
                components['outputs']['output_text'],
                components['outputs']['output_json'],
                components['outputs']['processing_log']
            ]
        )
        
        # 纯文本处理
        components['process_text_btn'].click(
            fn=self.process_text_input,
            inputs=[components['text_only_input']],
            outputs=[
                components['outputs']['output_text'],
                components['outputs']['output_json'],
                components['outputs']['processing_log']
            ]
        )
        
        # 清除按钮
        components['outputs']['clear_btn'].click(
            fn=lambda: ("", {}, "", None, []),
            outputs=[
                components['outputs']['output_text'],
                components['outputs']['output_json'],
                components['outputs']['processing_log'],
                components['outputs']['extracted_audio'],
                components['outputs']['extracted_images']
            ]
        )
    
    def process_video_input(self, video_file, prompt, extract_audio, extract_last_frame, optimization_preset):
        """处理视频输入"""
        if not self.current_model_loaded:
            return "❌ 模型未加载", {}, "错误：模型未加载", None, []
        
        if not video_file:
            return "❌ 请上传视频文件", {}, "错误：未上传视频文件", None, []
        
        try:
            with self.processing_lock:
                log_messages = []
                log_messages.append("🎬 开始处理视频...")
                
                # 优化视频（如果需要）
                handler = MemoryOptimizedVideoHandler()
                optimized_video, optimization_info = handler.auto_optimize_for_memory(video_file)
                
                if optimization_info.get('optimized'):
                    log_messages.append(f"✅ 视频已优化: {optimization_info}")
                
                # 处理视频
                video_results = self.video_processor.process_video_for_model(
                    optimized_video,
                    extract_audio=extract_audio,
                    extract_last_frame=extract_last_frame
                )
                
                if not video_results['success']:
                    error_msg = f"❌ 视频处理失败: {video_results.get('error', '未知错误')}"
                    return error_msg, video_results, '\n'.join(log_messages + [error_msg]), None, []
                
                log_messages.append("✅ 视频处理完成")
                log_messages.append("🤖 开始AI推理...")
                
                # AI推理
                system_prompt = "You are Qwen, a helpful AI assistant capable of analyzing videos, audio, and images."
                
                result = self.model_manager.inference(
                    "video",
                    video_path=optimized_video,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    extract_audio=extract_audio,
                    extract_last_frame=extract_last_frame
                )
                
                log_messages.append("✅ AI推理完成")
                
                # 准备输出
                extracted_images = []
                extracted_audio_file = None
                
                if video_results.get('frame_paths'):
                    extracted_images = video_results['frame_paths']
                
                if video_results.get('audio_path'):
                    extracted_audio_file = video_results['audio_path']
                
                output_data = {
                    'ai_result': result,
                    'video_info': video_results.get('video_info', {}),
                    'optimization_info': optimization_info,
                    'processing_success': True
                }
                
                return result, output_data, '\n'.join(log_messages), extracted_audio_file, extracted_images
                
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            logger.error(error_msg)
            return error_msg, {'error': str(e)}, error_msg, None, []
    
    def process_audio_image_input(self, audio_file, image_file, prompt):
        """处理音频+图片输入"""
        if not self.current_model_loaded:
            return "❌ 模型未加载", {}, "错误：模型未加载"
        
        if not audio_file or not image_file:
            return "❌ 请同时上传音频和图片文件", {}, "错误：缺少音频或图片文件"
        
        try:
            with self.processing_lock:
                log_messages = ["🎵 开始处理音频+图片..."]
                
                # 构建消息
                content = [
                    {"type": "audio", "audio": audio_file},
                    {"type": "image", "image": image_file}
                ]
                
                if prompt:
                    content.append({"type": "text", "text": prompt})
                
                messages = [
                    {
                        "role": "system", 
                        "content": [{"type": "text", "text": "You are Qwen, a helpful AI assistant."}]
                    },
                    {"role": "user", "content": content}
                ]
                
                log_messages.append("🤖 开始AI推理...")
                
                result = self.model_manager.inference("multimodal", messages=messages)
                
                log_messages.append("✅ 处理完成")
                
                output_data = {
                    'ai_result': result,
                    'input_type': 'audio_image',
                    'processing_success': True
                }
                
                return result, output_data, '\n'.join(log_messages)
                
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            logger.error(error_msg)
            return error_msg, {'error': str(e)}, error_msg
    
    def process_text_image_input(self, text, image_file):
        """处理文本+图片输入"""
        if not self.current_model_loaded:
            return "❌ 模型未加载", {}, "错误：模型未加载"
        
        if not text or not image_file:
            return "❌ 请输入文本并上传图片", {}, "错误：缺少文本或图片"
        
        try:
            with self.processing_lock:
                log_messages = ["📝 开始处理文本+图片..."]
                
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are Qwen, a helpful AI assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "image", "image": image_file}
                        ]
                    }
                ]
                
                log_messages.append("🤖 开始AI推理...")
                
                result = self.model_manager.inference("multimodal", messages=messages)
                
                log_messages.append("✅ 处理完成")
                
                output_data = {
                    'ai_result': result,
                    'input_type': 'text_image',
                    'processing_success': True
                }
                
                return result, output_data, '\n'.join(log_messages)
                
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            logger.error(error_msg)
            return error_msg, {'error': str(e)}, error_msg
    
    def process_text_input(self, text):
        """处理纯文本输入"""
        if not self.current_model_loaded:
            return "❌ 模型未加载", {}, "错误：模型未加载"
        
        if not text:
            return "❌ 请输入文本", {}, "错误：未输入文本"
        
        try:
            with self.processing_lock:
                log_messages = ["💬 开始处理文本..."]
                
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are Qwen, a helpful AI assistant."}]
                    },
                    {
                        "role": "user", 
                        "content": [{"type": "text", "text": text}]
                    }
                ]
                
                log_messages.append("🤖 开始AI推理...")
                
                result = self.model_manager.inference("text", messages=messages)
                
                log_messages.append("✅ 处理完成")
                
                output_data = {
                    'ai_result': result,
                    'input_type': 'text_only',
                    'processing_success': True
                }
                
                return result, output_data, '\n'.join(log_messages)
                
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            logger.error(error_msg)
            return error_msg, {'error': str(e)}, error_msg
    
    def get_current_memory_info(self) -> Dict[str, Any]:
        """获取当前内存信息"""
        try:
            if hasattr(self, 'memory_loader'):
                return self.memory_loader.memory_manager.get_system_memory_info()
            else:
                return {"status": "内存管理器未初始化"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_loaded": self.current_model_loaded,
            "model_path": os.getenv("MODEL_PATH", "未设置"),
            "available_presets": InferencePresets.list_presets(),
            "memory_presets": MemoryPresets.list_presets()
        }
    
    def launch(self, **kwargs):
        """启动应用"""
        interface = self.create_interface()
        
        # 设置默认参数
        default_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False,
            "debug": True,
            "show_error": True
        }
        default_kwargs.update(kwargs)
        
        try:
            logger.info("🚀 启动Gradio应用...")
            logger.info(f"📱 多模态AI推理平台")
            logger.info(f"🌐 访问地址: http://localhost:{default_kwargs['server_port']}")
            
            interface.launch(**default_kwargs)
            
        except Exception as e:
            logger.error(f"应用启动失败: {e}")
            raise
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        try:
            # 清理临时文件
            for temp_file in self.temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            
            # 清理模型
            if self.model_manager:
                self.model_manager.cleanup()
            
            # 清理视频处理器
            if hasattr(self.video_processor, 'cleanup'):
                self.video_processor.cleanup()
            
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")

def main():
    """主函数"""
    try:
        # 检查环境
        if not os.getenv("MODEL_PATH"):
            logger.warning("MODEL_PATH环境变量未设置，使用默认路径")
        
        # 创建应用
        app = GradioApp()
        
        # 启动应用
        app.launch()
        
    except KeyboardInterrupt:
        logger.info("用户中断，正在清理...")
    except Exception as e:
        logger.error(f"应用运行失败: {e}")
        raise

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态视频处理服务的Gradio界面
支持视频、音频+图片、文本+图片等多种输入方式
"""

import gradio as gr
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# 导入多模态视频处理服务
from multimodal_video_service import MultimodalVideoService, process_multimodal_input

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioMultimodalInterface:
    """Gradio多模态界面类"""
    
    def __init__(self):
        self.service = MultimodalVideoService()
        self.temp_files = []
        self.setup_interface()
    
    def setup_interface(self):
        """设置Gradio界面"""
        # 创建界面组件
        with gr.Blocks(
            title="多模态视频处理服务",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: 0 auto !important;
            }
            .input-section {
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                background: #f8f9fa;
            }
            .output-section {
                border: 2px solid #28a745;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                background: #f8fff9;
            }
            """
        ) as self.interface:
            
            gr.Markdown("""
            # 🎥 多模态视频处理服务
            
            支持多种输入方式的多模态内容处理，包括：
            - 🎬 视频文件（自动提取音频和关键帧）
            - 🎵 音频文件 + 图片
            - 📝 文本 + 图片
            
            输出：文本描述和分析结果
            """)
            
            # 输入方式选择
            with gr.Row():
                input_method = gr.Radio(
                    choices=["视频文件", "音频+图片", "文本+图片"],
                    value="视频文件",
                    label="选择输入方式",
                    info="请选择您要使用的输入方式"
                )
            
            # 视频输入部分
            with gr.Group(visible=True) as video_input_group:
                gr.Markdown("### 🎬 视频文件输入")
                with gr.Row():
                    video_file = gr.Video(
                        label="上传视频文件",
                        info="支持MP4, AVI, MOV, MKV等格式",
                        height=300
                    )
                    video_preview = gr.Video(
                        label="视频预览",
                        height=300,
                        interactive=False
                    )
                
                with gr.Row():
                    process_video_btn = gr.Button(
                        "🚀 处理视频",
                        variant="primary",
                        size="lg"
                    )
                    clear_video_btn = gr.Button("🗑️ 清除", variant="secondary")
            
            # 音频+图片输入部分
            with gr.Group(visible=False) as audio_image_group:
                gr.Markdown("### 🎵 音频+图片输入")
                with gr.Row():
                    audio_file = gr.Audio(
                        label="上传音频文件",
                        info="支持WAV, MP3, FLAC等格式",
                        type="filepath"
                    )
                    image_file_audio = gr.Image(
                        label="上传图片",
                        info="支持JPG, PNG, BMP等格式",
                        height=300
                    )
                
                with gr.Row():
                    process_audio_image_btn = gr.Button(
                        "🚀 处理音频+图片",
                        variant="primary",
                        size="lg"
                    )
                    clear_audio_image_btn = gr.Button("🗑️ 清除", variant="secondary")
            
            # 文本+图片输入部分
            with gr.Group(visible=False) as text_image_group:
                gr.Markdown("### 📝 文本+图片输入")
                with gr.Row():
                    text_input = gr.Textbox(
                        label="输入文本",
                        placeholder="请输入您要分析的文本内容...",
                        lines=4,
                        max_lines=10
                    )
                    image_file_text = gr.Image(
                        label="上传图片",
                        info="支持JPG, PNG, BMP等格式",
                        height=300
                    )
                
                with gr.Row():
                    process_text_image_btn = gr.Button(
                        "🚀 处理文本+图片",
                        variant="primary",
                        size="lg"
                    )
                    clear_text_image_btn = gr.Button("🗑️ 清除", variant="secondary")
            
            # 输出部分
            with gr.Group() as output_group:
                gr.Markdown("### 📊 处理结果")
                
                with gr.Row():
                    result_text = gr.Textbox(
                        label="处理结果",
                        lines=10,
                        max_lines=20,
                        interactive=False,
                        placeholder="处理结果将在这里显示..."
                    )
                
                with gr.Row():
                    result_json = gr.JSON(
                        label="详细结果 (JSON)",
                        visible=True
                    )
                
                with gr.Row():
                    download_btn = gr.Button("💾 下载结果", variant="secondary")
                    clear_output_btn = gr.Button("🗑️ 清除输出", variant="secondary")
            
            # 状态信息
            with gr.Group():
                gr.Markdown("### ℹ️ 服务信息")
                with gr.Row():
                    status_text = gr.Textbox(
                        label="服务状态",
                        value="🟢 服务运行正常",
                        interactive=False
                    )
                    supported_formats = gr.JSON(
                        label="支持的格式",
                        value=self.service.get_supported_formats(),
                        interactive=False
                    )
            
            # 事件处理
            self.setup_event_handlers(
                input_method, video_input_group, audio_image_group, text_image_group,
                video_file, video_preview, audio_file, image_file_audio,
                text_input, image_file_text, result_text, result_json,
                process_video_btn, process_audio_image_btn, process_text_image_btn,
                clear_video_btn, clear_audio_image_btn, clear_text_image_btn,
                clear_output_btn, download_btn, status_text
            )
    
    def setup_event_handlers(self, input_method, video_input_group, audio_image_group, 
                           text_image_group, video_file, video_preview, audio_file, 
                           image_file_audio, text_input, image_file_text, result_text, 
                           result_json, process_video_btn, process_audio_image_btn, 
                           process_text_image_btn, clear_video_btn, clear_audio_image_btn, 
                           clear_text_image_btn, clear_output_btn, download_btn, status_text):
        """设置事件处理器"""
        
        # 输入方式切换
        def on_input_method_change(method):
            if method == "视频文件":
                return gr.Group(visible=True), gr.Group(visible=False), gr.Group(visible=False)
            elif method == "音频+图片":
                return gr.Group(visible=False), gr.Group(visible=True), gr.Group(visible=False)
            else:  # 文本+图片
                return gr.Group(visible=False), gr.Group(visible=False), gr.Group(visible=True)
        
        input_method.change(
            fn=on_input_method_change,
            inputs=[input_method],
            outputs=[video_input_group, audio_image_group, text_image_group]
        )
        
        # 视频预览
        def on_video_change(video):
            if video:
                return video
            return None
        
        video_file.change(
            fn=on_video_change,
            inputs=[video_file],
            outputs=[video_preview]
        )
        
        # 处理视频
        def process_video(video):
            if not video:
                return "❌ 请先上传视频文件", {}, "❌ 处理失败：未上传视频文件"
            
            try:
                status_text.update("🔄 正在处理视频...")
                result = process_multimodal_input(video_file=video)
                
                if result["success"]:
                    status_text.update("✅ 视频处理完成")
                    return result["result"], result, "✅ 处理成功"
                else:
                    status_text.update("❌ 视频处理失败")
                    return f"❌ 处理失败：{result['error']}", result, "❌ 处理失败"
                    
            except Exception as e:
                error_msg = f"❌ 处理异常：{str(e)}"
                status_text.update(error_msg)
                return error_msg, {"error": str(e)}, error_msg
        
        process_video_btn.click(
            fn=process_video,
            inputs=[video_file],
            outputs=[result_text, result_json, status_text]
        )
        
        # 处理音频+图片
        def process_audio_image(audio, image):
            if not audio or not image:
                return "❌ 请同时上传音频和图片文件", {}, "❌ 处理失败：缺少音频或图片文件"
            
            try:
                status_text.update("🔄 正在处理音频+图片...")
                result = process_multimodal_input(audio_file=audio, image_file=image)
                
                if result["success"]:
                    status_text.update("✅ 音频+图片处理完成")
                    return result["result"], result, "✅ 处理成功"
                else:
                    status_text.update("❌ 音频+图片处理失败")
                    return f"❌ 处理失败：{result['error']}", result, "❌ 处理失败"
                    
            except Exception as e:
                error_msg = f"❌ 处理异常：{str(e)}"
                status_text.update(error_msg)
                return error_msg, {"error": str(e)}, error_msg
        
        process_audio_image_btn.click(
            fn=process_audio_image,
            inputs=[audio_file, image_file_audio],
            outputs=[result_text, result_json, status_text]
        )
        
        # 处理文本+图片
        def process_text_image(text, image):
            if not text or not image:
                return "❌ 请输入文本并上传图片", {}, "❌ 处理失败：缺少文本或图片"
            
            try:
                status_text.update("🔄 正在处理文本+图片...")
                result = process_multimodal_input(text_input=text, image_file=image)
                
                if result["success"]:
                    status_text.update("✅ 文本+图片处理完成")
                    return result["result"], result, "✅ 处理成功"
                else:
                    status_text.update("❌ 文本+图片处理失败")
                    return f"❌ 处理失败：{result['error']}", result, "❌ 处理失败"
                    
            except Exception as e:
                error_msg = f"❌ 处理异常：{str(e)}"
                status_text.update(error_msg)
                return error_msg, {"error": str(e)}, error_msg
        
        process_text_image_btn.click(
            fn=process_text_image,
            inputs=[text_input, image_file_text],
            outputs=[result_text, result_json, status_text]
        )
        
        # 清除功能
        def clear_video():
            return None, None
        
        def clear_audio_image():
            return None, None
        
        def clear_text_image():
            return "", None
        
        def clear_output():
            return "", {}, "🟢 服务运行正常"
        
        clear_video_btn.click(
            fn=clear_video,
            outputs=[video_file, video_preview]
        )
        
        clear_audio_image_btn.click(
            fn=clear_audio_image,
            outputs=[audio_file, image_file_audio]
        )
        
        clear_text_image_btn.click(
            fn=clear_text_image,
            outputs=[text_input, image_file_text]
        )
        
        clear_output_btn.click(
            fn=clear_output,
            outputs=[result_text, result_json, status_text]
        )
        
        # 下载结果
        def download_result(result_json_data):
            if not result_json_data:
                return None
            
            try:
                # 创建临时文件
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.json',
                    delete=False,
                    encoding='utf-8'
                )
                
                json.dump(result_json_data, temp_file, ensure_ascii=False, indent=2)
                temp_file.close()
                
                self.temp_files.append(temp_file.name)
                return temp_file.name
                
            except Exception as e:
                logger.error(f"下载文件创建失败: {str(e)}")
                return None
        
        download_btn.click(
            fn=download_result,
            inputs=[result_json],
            outputs=[gr.File(label="下载结果")]
        )
    
    def launch(self, **kwargs):
        """启动Gradio界面"""
        try:
            self.interface.launch(**kwargs)
        except Exception as e:
            logger.error(f"界面启动失败: {str(e)}")
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
            
            # 清理服务
            self.service.cleanup()
            
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {str(e)}")

def main():
    """主函数"""
    try:
        # 创建并启动界面
        interface = GradioMultimodalInterface()
        
        print("🚀 启动多模态视频处理服务界面...")
        print("📱 支持多种输入方式：视频、音频+图片、文本+图片")
        print("🌐 界面将在浏览器中打开")
        
        # 启动界面
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 服务启动失败: {str(e)}")
        logger.error(f"服务启动失败: {str(e)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态视频处理服务测试脚本
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_multimodal_service():
    """测试多模态视频处理服务"""
    try:
        # 导入服务
        from multimodal_video_service import MultimodalVideoService
        
        logger.info("🧪 开始测试多模态视频处理服务...")
        
        # 创建服务实例
        service = MultimodalVideoService()
        
        # 测试1: 文本+图片输入
        logger.info("📝 测试文本+图片输入...")
        test_image = "test_image.png"
        
        if os.path.exists(test_image):
            result = service.process_input(
                text_input="这是一张测试图片，请分析其内容",
                image_file=test_image
            )
            logger.info(f"文本+图片测试结果: {result}")
        else:
            logger.warning(f"测试图片 {test_image} 不存在，跳过此测试")
        
        # 测试2: 获取支持的格式
        logger.info("📋 测试获取支持的格式...")
        formats = service.get_supported_formats()
        logger.info(f"支持的格式: {formats}")
        
        # 测试3: 音频+图片输入（如果有测试音频）
        test_audio = "test_video_audio.wav"
        if os.path.exists(test_audio) and os.path.exists(test_image):
            logger.info("🎵 测试音频+图片输入...")
            result = service.process_input(
                audio_file=test_audio,
                image_file=test_image
            )
            logger.info(f"音频+图片测试结果: {result}")
        else:
            logger.warning("测试音频文件不存在，跳过音频+图片测试")
        
        # 测试4: 视频输入（如果有测试视频）
        test_video = "test_video.mp4"
        if os.path.exists(test_video):
            logger.info("🎬 测试视频输入...")
            result = service.process_input(video_file=test_video)
            logger.info(f"视频测试结果: {result}")
        else:
            logger.warning("测试视频文件不存在，跳过视频测试")
        
        logger.info("✅ 所有测试完成")
        
        # 清理资源
        service.cleanup()
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        raise

def test_gradio_interface():
    """测试Gradio界面"""
    try:
        logger.info("🖥️ 测试Gradio界面...")
        
        # 导入界面
        from gradio_multimodal_interface import GradioMultimodalInterface
        
        # 创建界面实例
        interface = GradioMultimodalInterface()
        
        logger.info("✅ Gradio界面创建成功")
        
        # 清理资源
        interface.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Gradio界面测试失败: {str(e)}")
        raise

def main():
    """主测试函数"""
    try:
        logger.info("🚀 开始多模态视频处理服务测试...")
        
        # 测试核心服务
        test_multimodal_service()
        
        # 测试Gradio界面
        test_gradio_interface()
        
        logger.info("🎉 所有测试通过！")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

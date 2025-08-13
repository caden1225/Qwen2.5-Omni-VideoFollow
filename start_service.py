#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态视频处理服务启动脚本
"""

import os
import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查依赖"""
    required_packages = [
        'torch', 'numpy', 'PIL', 'librosa', 'soundfile', 'cv2', 'gradio'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"缺少以下依赖包: {', '.join(missing_packages)}")
        logger.info("请运行: pip install -r requirements_multimodal.txt")
        return False
    
    return True

def main():
    """主函数"""
    try:
        logger.info("🚀 启动多模态视频处理服务...")
        
        # 检查依赖
        if not check_dependencies():
            sys.exit(1)
        
        # 导入并启动服务
        from gradio_multimodal_interface import main as launch_interface
        
        logger.info("✅ 依赖检查通过")
        logger.info("📱 启动Gradio界面...")
        
        # 启动界面
        launch_interface()
        
    except KeyboardInterrupt:
        logger.info("👋 服务已停止")
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的应用启动脚本
快速启动Qwen2.5-Omni多模态推理服务
"""

import os
import sys
import logging
from pathlib import Path

# 设置项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

def setup_basic_environment():
    """设置基础环境"""
    # 设置模型路径
    model_path = os.getenv('MODEL_PATH', '/home/caden/workplace/models/Qwen2.5-Omni-3B')
    os.environ['MODEL_PATH'] = model_path
    
    # 设置CUDA环境
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 创建必要目录
    (project_root / "outputs").mkdir(exist_ok=True)
    (project_root / "extracted_media").mkdir(exist_ok=True)
    (project_root / "extracted_media" / "audio").mkdir(exist_ok=True)
    (project_root / "extracted_media" / "frames").mkdir(exist_ok=True)
    
    print(f"✅ 环境设置完成")
    print(f"📁 模型路径: {model_path}")
    print(f"📁 工作目录: {project_root}")

def check_dependencies():
    """检查关键依赖"""
    required_modules = ['torch', 'transformers', 'gradio', 'librosa', 'cv2', 'PIL']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing.append(module)
            print(f"❌ {module}")
    
    if missing:
        print(f"\n⚠️ 缺少依赖: {', '.join(missing)}")
        print(f"请运行: pip install -r requirements_integrated.txt")
        return False
    
    return True

def quick_start():
    """快速启动"""
    print("🚀 Qwen2.5-Omni 多模态推理平台")
    print("="*50)
    
    # 设置环境
    setup_basic_environment()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 导入并启动应用
    try:
        print("\n📥 导入应用模块...")
        from gradio_app import GradioApp
        
        print("🔧 初始化应用...")
        app = GradioApp()
        
        print("🌐 启动Web界面...")
        print("访问地址: http://localhost:7860")
        print("按 Ctrl+C 停止服务\n")
        
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False
        )
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("请检查项目文件是否完整")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        quick_start()
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        sys.exit(1)
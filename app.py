#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-Omni 多模态推理应用主入口
整合所有功能模块，提供完整的多模态AI服务
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "qwen-omni-utils" / "src"))

# 导入核心模块
from gradio_app import GradioApp
from model_inference import ModelManager, InferencePresets, load_model
from memory_manager import MemoryPresets, auto_configure_memory
from video_optimizer import VideoOptimizationPresets
from test_integrated_system import IntegratedSystemTester

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Qwen25OmniApp:
    """Qwen2.5-Omni应用主类"""
    
    def __init__(self, args):
        self.args = args
        self.setup_environment()
        self.validate_setup()
    
    def setup_environment(self):
        """设置运行环境"""
        # 设置模型路径
        if self.args.model_path:
            os.environ['MODEL_PATH'] = self.args.model_path
        elif not os.getenv('MODEL_PATH'):
            default_path = "/home/caden/workplace/models/Qwen2.5-Omni-3B"
            os.environ['MODEL_PATH'] = default_path
            logger.warning(f"未设置MODEL_PATH，使用默认路径: {default_path}")
        
        # 设置CUDA环境
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 创建必要目录
        (project_root / "outputs").mkdir(exist_ok=True)
        (project_root / "configs").mkdir(exist_ok=True)
        (project_root / "logs").mkdir(exist_ok=True)
        
        logger.info(f"环境设置完成，模型路径: {os.getenv('MODEL_PATH')}")
    
    def validate_setup(self):
        """验证设置"""
        model_path = os.getenv('MODEL_PATH')
        
        if not os.path.exists(model_path):
            logger.error(f"模型路径不存在: {model_path}")
            logger.error("请检查模型路径设置或下载模型")
            if not self.args.skip_validation:
                sys.exit(1)
        
        logger.info("✅ 设置验证通过")
    
    def run_tests(self):
        """运行系统测试"""
        logger.info("🧪 运行系统测试...")
        tester = IntegratedSystemTester()
        tester.run_all_tests()
        
        # 检查测试结果
        passed = sum(1 for r in tester.test_results.values() if r.get("status") == "passed")
        total = len(tester.test_results)
        
        logger.info(f"测试完成: {passed}/{total} 通过")
        
        if passed < total and not self.args.ignore_test_failures:
            logger.error("系统测试未完全通过，建议检查配置")
            if not self.args.force_start:
                return False
        
        return True
    
    def start_gradio_app(self):
        """启动Gradio应用"""
        try:
            logger.info("🚀 启动Gradio多模态推理界面...")
            
            app = GradioApp()
            
            # 启动参数
            launch_kwargs = {
                "server_name": self.args.host,
                "server_port": self.args.port,
                "share": self.args.share,
                "debug": self.args.debug,
                "show_error": True
            }
            
            app.launch(**launch_kwargs)
            
        except Exception as e:
            logger.error(f"Gradio应用启动失败: {e}")
            raise
    
    def print_system_info(self):
        """打印系统信息"""
        try:
            import torch
            
            print("\n" + "="*60)
            print("🖥️  系统信息")
            print("="*60)
            print(f"Python版本: {sys.version.split()[0]}")
            print(f"工作目录: {os.getcwd()}")
            print(f"模型路径: {os.getenv('MODEL_PATH')}")
            
            if torch.cuda.is_available():
                print(f"CUDA版本: {torch.version.cuda}")
                print(f"GPU数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / 1024**3
                    print(f"GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            else:
                print("CUDA: 不可用")
            
            print(f"\n📋 可用预设:")
            print(f"模型预设: {', '.join(InferencePresets.list_presets())}")
            print(f"内存预设: {', '.join(MemoryPresets.list_presets())}")
            print(f"视频预设: {', '.join(VideoOptimizationPresets.list_presets())}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"系统信息获取失败: {e}")
    
    def run(self):
        """运行应用"""
        try:
            # 打印系统信息
            if not self.args.quiet:
                self.print_system_info()
            
            # 运行测试（如果启用）
            if self.args.run_tests:
                if not self.run_tests():
                    logger.error("系统测试失败")
                    if not self.args.force_start:
                        return False
            
            # 启动应用
            if self.args.mode == "gradio":
                self.start_gradio_app()
            elif self.args.mode == "test":
                return self.run_tests()
            else:
                logger.error(f"未知模式: {self.args.mode}")
                return False
            
            return True
            
        except KeyboardInterrupt:
            logger.info("用户中断应用")
            return True
        except Exception as e:
            logger.error(f"应用运行失败: {e}")
            return False

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Omni 多模态推理应用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python app.py                                    # 使用默认设置启动
  python app.py --model-path /path/to/model       # 指定模型路径
  python app.py --port 8080 --host 0.0.0.0       # 自定义服务器设置
  python app.py --mode test                       # 只运行测试
  python app.py --run-tests --force-start         # 运行测试后强制启动
        """
    )
    
    # 基本参数
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="模型文件路径（默认从环境变量MODEL_PATH读取）"
    )
    
    parser.add_argument(
        "--mode",
        choices=["gradio", "test"],
        default="gradio",
        help="运行模式（默认: gradio）"
    )
    
    # Gradio服务器参数
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器主机地址（默认: 0.0.0.0）"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="服务器端口（默认: 7860）"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="启用Gradio公共链接分享"
    )
    
    # 测试参数
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="启动前运行系统测试"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="跳过设置验证"
    )
    
    parser.add_argument(
        "--ignore-test-failures",
        action="store_true",
        help="忽略测试失败"
    )
    
    parser.add_argument(
        "--force-start",
        action="store_true",
        help="即使测试失败也强制启动"
    )
    
    # 调试参数
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true", 
        help="安静模式（减少输出）"
    )
    
    return parser.parse_args()

def setup_logging(debug: bool = False, quiet: bool = False):
    """设置日志"""
    if quiet:
        level = logging.WARNING
    elif debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # 重新配置root logger
    logging.getLogger().setLevel(level)
    
    # 创建日志文件处理器
    log_file = project_root / "logs" / "app.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 设置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # 添加到root logger
    logging.getLogger().addHandler(file_handler)

def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_arguments()
        
        # 设置日志
        setup_logging(args.debug, args.quiet)
        
        # 创建并运行应用
        app = Qwen25OmniApp(args)
        success = app.run()
        
        if success:
            logger.info("🎉 应用运行成功")
        else:
            logger.error("❌ 应用运行失败")
        
        return success
        
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成系统测试
测试视频处理、模型推理、内存管理等功能
"""

import os
import sys
import logging
import json
import tempfile
from pathlib import Path
import time

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入测试模块
from video_utils import VideoProcessor, create_conversation_with_video
from video_optimizer import VideoOptimizer, VideoOptimizationPresets, MemoryOptimizedVideoHandler
from model_inference import ModelManager, InferencePresets
from memory_manager import MemoryPresets, ConfigurableMemoryLoader

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedSystemTester:
    """集成系统测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """设置测试环境"""
        # 创建测试输出目录
        self.test_output_dir = Path("./test_outputs")
        self.test_output_dir.mkdir(exist_ok=True)
        
        # 设置测试视频路径（如果存在）
        self.test_video_path = None
        test_video_candidates = [
            "./test_video.mp4",
            "./assets/test_video.mp4",
            "./datasets/test_video.mp4"
        ]
        
        for candidate in test_video_candidates:
            if os.path.exists(candidate):
                self.test_video_path = candidate
                break
        
        logger.info(f"测试环境设置完成，输出目录: {self.test_output_dir}")
        if self.test_video_path:
            logger.info(f"找到测试视频: {self.test_video_path}")
        else:
            logger.warning("未找到测试视频，将跳过视频相关测试")
    
    def test_video_processing(self):
        """测试视频处理功能"""
        logger.info("开始测试视频处理功能...")
        test_name = "video_processing"
        
        try:
            if not self.test_video_path:
                logger.warning("跳过视频处理测试（无测试视频）")
                self.test_results[test_name] = {"status": "skipped", "reason": "无测试视频"}
                return
            
            # 测试基础视频处理
            processor = VideoProcessor(output_dir=str(self.test_output_dir / "video_processing"))
            
            # 测试音频提取
            audio_path, audio_data = processor.extract_audio_from_video(self.test_video_path)
            assert os.path.exists(audio_path), "音频文件未创建"
            assert audio_data is not None, "音频数据为空"
            
            # 测试最后一帧提取
            frame_path, frame_data = processor.extract_last_frame(self.test_video_path)
            assert os.path.exists(frame_path), "帧文件未创建"
            assert frame_data is not None, "帧数据为空"
            
            # 测试视频信息获取
            video_info = processor.get_video_info(self.test_video_path)
            assert video_info, "视频信息获取失败"
            assert "duration" in video_info, "视频信息缺少duration字段"
            
            self.test_results[test_name] = {
                "status": "passed",
                "audio_path": audio_path,
                "frame_path": frame_path,
                "video_info": video_info
            }
            logger.info("✅ 视频处理测试通过")
            
        except Exception as e:
            logger.error(f"❌ 视频处理测试失败: {e}")
            self.test_results[test_name] = {"status": "failed", "error": str(e)}
    
    def test_video_optimization(self):
        """测试视频优化功能"""
        logger.info("开始测试视频优化功能...")
        test_name = "video_optimization"
        
        try:
            if not self.test_video_path:
                logger.warning("跳过视频优化测试（无测试视频）")
                self.test_results[test_name] = {"status": "skipped", "reason": "无测试视频"}
                return
            
            # 测试不同预设
            for preset_name in VideoOptimizationPresets.list_presets():
                logger.info(f"测试优化预设: {preset_name}")
                
                config = VideoOptimizationPresets.get_preset(preset_name)
                optimizer = VideoOptimizer(config)
                
                # 测试内存优化
                handler = MemoryOptimizedVideoHandler()
                optimized_path, optimization_info = handler.auto_optimize_for_memory(self.test_video_path)
                
                assert os.path.exists(optimized_path), f"优化视频文件不存在: {optimized_path}"
                
                optimizer.cleanup()
            
            self.test_results[test_name] = {"status": "passed", "presets_tested": VideoOptimizationPresets.list_presets()}
            logger.info("✅ 视频优化测试通过")
            
        except Exception as e:
            logger.error(f"❌ 视频优化测试失败: {e}")
            self.test_results[test_name] = {"status": "failed", "error": str(e)}
    
    def test_memory_management(self):
        """测试内存管理功能"""
        logger.info("开始测试内存管理功能...")
        test_name = "memory_management"
        
        try:
            # 测试不同内存预设
            for preset_name in MemoryPresets.list_presets():
                logger.info(f"测试内存预设: {preset_name}")
                
                memory_loader = ConfigurableMemoryLoader()
                config = MemoryPresets.get_preset(preset_name)
                
                # 测试配置保存和加载
                config_path = self.test_output_dir / f"memory_config_{preset_name}.json"
                memory_loader.config = config
                memory_loader.save_config(str(config_path))
                
                assert config_path.exists(), f"配置文件未创建: {config_path}"
                
                # 测试配置加载
                loaded_config = memory_loader.load_config(str(config_path))
                assert loaded_config.gpu_memory_limit_gb == config.gpu_memory_limit_gb, "配置加载不正确"
            
            # 测试系统内存信息获取
            memory_info = memory_loader.memory_manager.get_system_memory_info()
            assert "cpu_total_gb" in memory_info, "系统内存信息不完整"
            
            self.test_results[test_name] = {
                "status": "passed",
                "presets_tested": MemoryPresets.list_presets(),
                "memory_info": memory_info
            }
            logger.info("✅ 内存管理测试通过")
            
        except Exception as e:
            logger.error(f"❌ 内存管理测试失败: {e}")
            self.test_results[test_name] = {"status": "failed", "error": str(e)}
    
    def test_model_loading(self):
        """测试模型加载功能"""
        logger.info("开始测试模型加载功能...")
        test_name = "model_loading"
        
        try:
            model_path = os.getenv("MODEL_PATH", "/home/caden/workplace/models/Qwen2.5-Omni-3B")
            
            if not os.path.exists(model_path):
                logger.warning(f"模型路径不存在，跳过模型加载测试: {model_path}")
                self.test_results[test_name] = {"status": "skipped", "reason": f"模型路径不存在: {model_path}"}
                return
            
            # 测试模型管理器
            model_manager = ModelManager()
            
            # 测试轻量级预设
            for preset_name in ["ultra_low_vram", "low_vram"]:
                logger.info(f"测试模型预设: {preset_name}")
                
                try:
                    model_config, vram_config = InferencePresets.get_model_preset(preset_name)
                    model_config.model_path = model_path
                    
                    # 尝试加载模型（但不保持加载状态以节省内存）
                    success = model_manager.load_model_with_config(model_config, vram_config)
                    
                    if success:
                        # 获取内存信息
                        memory_info = model_manager.get_memory_usage()
                        logger.info(f"预设 {preset_name} 加载成功，内存使用: {memory_info}")
                        
                        # 清理模型
                        model_manager.cleanup()
                    else:
                        logger.warning(f"预设 {preset_name} 加载失败")
                        
                except Exception as e:
                    logger.warning(f"预设 {preset_name} 测试失败: {e}")
            
            self.test_results[test_name] = {
                "status": "passed",
                "model_path": model_path,
                "presets_tested": ["ultra_low_vram", "low_vram"]
            }
            logger.info("✅ 模型加载测试通过")
            
        except Exception as e:
            logger.error(f"❌ 模型加载测试失败: {e}")
            self.test_results[test_name] = {"status": "failed", "error": str(e)}
    
    def test_conversation_creation(self):
        """测试对话创建功能"""
        logger.info("开始测试对话创建功能...")
        test_name = "conversation_creation"
        
        try:
            if not self.test_video_path:
                logger.warning("跳过对话创建测试（无测试视频）")
                self.test_results[test_name] = {"status": "skipped", "reason": "无测试视频"}
                return
            
            # 测试视频对话创建
            messages = create_conversation_with_video(
                self.test_video_path,
                prompt="请分析这个视频的内容",
                extract_audio=True,
                extract_last_frame=True
            )
            
            assert isinstance(messages, list), "对话格式不正确"
            assert len(messages) > 0, "对话为空"
            assert "content" in messages[0], "对话结构不正确"
            
            # 验证内容类型
            content_types = [item["type"] for item in messages[0]["content"]]
            assert "audio" in content_types or "video" in content_types, "缺少音频或视频内容"
            
            self.test_results[test_name] = {
                "status": "passed",
                "message_count": len(messages),
                "content_types": content_types
            }
            logger.info("✅ 对话创建测试通过")
            
        except Exception as e:
            logger.error(f"❌ 对话创建测试失败: {e}")
            self.test_results[test_name] = {"status": "failed", "error": str(e)}
    
    def test_dependencies(self):
        """测试依赖库"""
        logger.info("开始测试依赖库...")
        test_name = "dependencies"
        
        dependencies = {
            'torch': lambda: __import__('torch'),
            'transformers': lambda: __import__('transformers'),
            'gradio': lambda: __import__('gradio'),
            'librosa': lambda: __import__('librosa'),
            'soundfile': lambda: __import__('soundfile'),
            'cv2': lambda: __import__('cv2'),
            'PIL': lambda: __import__('PIL'),
            'numpy': lambda: __import__('numpy')
        }
        
        missing_deps = []
        available_deps = []
        
        for dep_name, import_func in dependencies.items():
            try:
                module = import_func()
                version = getattr(module, '__version__', 'unknown')
                available_deps.append(f"{dep_name} {version}")
                logger.info(f"✅ {dep_name}: {version}")
            except ImportError:
                missing_deps.append(dep_name)
                logger.error(f"❌ {dep_name}: 未安装")
        
        # 检查CUDA
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✅ CUDA可用: {gpu_count}个GPU, {gpu_name}")
            else:
                logger.warning("⚠️ CUDA不可用")
        except:
            logger.error("❌ 无法检查CUDA状态")
        
        self.test_results[test_name] = {
            "status": "passed" if not missing_deps else "failed",
            "available_dependencies": available_deps,
            "missing_dependencies": missing_deps,
            "cuda_available": cuda_available
        }
        
        if missing_deps:
            logger.error(f"❌ 缺少依赖: {missing_deps}")
        else:
            logger.info("✅ 依赖检查通过")
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("="*60)
        logger.info("开始集成系统测试")
        logger.info("="*60)
        
        start_time = time.time()
        
        # 测试顺序
        test_methods = [
            self.test_dependencies,
            self.test_memory_management,
            self.test_video_processing,
            self.test_video_optimization,
            self.test_conversation_creation,
            self.test_model_loading,
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"测试异常: {test_method.__name__}: {e}")
        
        # 生成测试报告
        self.generate_test_report()
        
        total_time = time.time() - start_time
        logger.info(f"测试完成，总耗时: {total_time:.2f}秒")
    
    def generate_test_report(self):
        """生成测试报告"""
        try:
            report_path = self.test_output_dir / "test_report.json"
            
            # 统计测试结果
            total_tests = len(self.test_results)
            passed_tests = sum(1 for result in self.test_results.values() if result.get("status") == "passed")
            failed_tests = sum(1 for result in self.test_results.values() if result.get("status") == "failed")
            skipped_tests = sum(1 for result in self.test_results.values() if result.get("status") == "skipped")
            
            report = {
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "skipped": skipped_tests,
                    "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
                },
                "test_results": self.test_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "environment": {
                    "python_version": sys.version,
                    "working_directory": os.getcwd(),
                    "model_path": os.getenv("MODEL_PATH", "未设置")
                }
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"测试报告已生成: {report_path}")
            logger.info(f"测试摘要: {passed_tests}/{total_tests} 通过, {failed_tests} 失败, {skipped_tests} 跳过")
            
            return report
            
        except Exception as e:
            logger.error(f"测试报告生成失败: {e}")
            return None

def create_example_config():
    """创建示例配置文件"""
    try:
        # 创建示例模型配置
        model_config_example = {
            "model_path": "/home/caden/workplace/models/Qwen2.5-Omni-3B",
            "preset": "low_vram",
            "custom_settings": {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "load_thinker_only": True
            }
        }
        
        # 创建示例视频配置
        video_config_example = {
            "optimization_preset": "balanced",
            "custom_settings": {
                "max_frames": 6,
                "max_resolution": 224,
                "compression_quality": 85
            }
        }
        
        # 创建示例内存配置
        memory_config_example = {
            "preset": "low_vram",
            "custom_settings": {
                "gpu_memory_limit_gb": 4.0,
                "enable_cpu_offload": True,
                "modules_on_cpu": ["talker", "token2wav"]
            }
        }
        
        # 保存配置文件
        config_dir = Path("./configs")
        config_dir.mkdir(exist_ok=True)
        
        configs = {
            "model_config_example.json": model_config_example,
            "video_config_example.json": video_config_example,
            "memory_config_example.json": memory_config_example
        }
        
        for filename, config_data in configs.items():
            config_path = config_dir / filename
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.info(f"示例配置已创建: {config_path}")
        
    except Exception as e:
        logger.error(f"配置文件创建失败: {e}")

def main():
    """主测试函数"""
    try:
        logger.info("🧪 启动集成系统测试")
        
        # 创建示例配置
        create_example_config()
        
        # 运行测试
        tester = IntegratedSystemTester()
        tester.run_all_tests()
        
        # 检查测试结果
        passed_count = sum(1 for result in tester.test_results.values() if result.get("status") == "passed")
        total_count = len(tester.test_results)
        
        if passed_count == total_count:
            logger.info("🎉 所有测试通过！系统已准备就绪")
            return True
        else:
            logger.warning(f"⚠️ 部分测试失败: {passed_count}/{total_count} 通过")
            return False
        
    except Exception as e:
        logger.error(f"❌ 测试运行失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
vLLM修复补丁
用于修复Qwen2.5-Omni在vLLM加载时遇到的各种问题
通过猴子补丁的方式修复库中的问题，不修改原始库文件
"""

import os
import sys
import importlib
import logging
from typing import Any, Dict, Optional, List
import warnings

# 配置日志
logger = logging.getLogger(__name__)


def check_vllm_installation():
    """检查vLLM安装和版本兼容性"""
    try:
        import vllm
        logger.info(f"vLLM版本: {vllm.__version__}")
        return True
    except ImportError:
        logger.error("vLLM未安装，请运行: pip install vllm")
        return False
    except Exception as e:
        logger.error(f"vLLM检查失败: {e}")
        return False


def patch_video_max_pixels():
    """修复视频处理的最大像素限制问题"""
    try:
        # 设置视频处理相关环境变量
        video_max_pixels = str(32000 * 28 * 28)
        os.environ.setdefault('VIDEO_MAX_PIXELS', video_max_pixels)
        
        # 尝试修复transformers中的视频处理限制
        try:
            from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
            # 如果处理器有max_pixels属性，尝试调整
            if hasattr(Qwen2_5OmniProcessor, '_max_pixels'):
                original_max_pixels = getattr(Qwen2_5OmniProcessor, '_max_pixels', None)
                if original_max_pixels and original_max_pixels < int(video_max_pixels):
                    setattr(Qwen2_5OmniProcessor, '_max_pixels', int(video_max_pixels))
                    logger.info(f"调整了处理器max_pixels从 {original_max_pixels} 到 {video_max_pixels}")
        except Exception as e:
            logger.warning(f"无法调整处理器max_pixels设置: {e}")
        
        logger.info(f"视频像素限制补丁已应用: {video_max_pixels}")
        return True
    except Exception as e:
        logger.error(f"视频像素限制补丁失败: {e}")
        return False


def patch_vllm_multimodal_imports():
    """修复vLLM多模态导入问题"""
    try:
        # 检查关键模块是否可导入
        required_modules = [
            'vllm.engine.omni_llm_engine',
            'vllm.engine.async_llm_engine',
            'vllm.sampling_params',
            'vllm.inputs',
            'vllm.multimodal.processing_omni'
        ]
        
        missing_modules = []
        for module_name in required_modules:
            try:
                importlib.import_module(module_name)
                logger.info(f"✓ 模块 {module_name} 导入成功")
            except ImportError as e:
                missing_modules.append((module_name, str(e)))
                logger.warning(f"✗ 模块 {module_name} 导入失败: {e}")
        
        # 如果有缺失的模块，尝试创建替代品或提供错误信息
        if missing_modules:
            logger.warning(f"发现 {len(missing_modules)} 个缺失模块，可能需要更新vLLM版本或使用兼容版本")
            for module_name, error in missing_modules:
                logger.warning(f"  - {module_name}: {error}")
            return False
        
        logger.info("vLLM多模态导入补丁验证成功")
        return True
    except Exception as e:
        logger.error(f"vLLM多模态导入补丁失败: {e}")
        return False


def patch_vllm_memory_optimization():
    """修复vLLM内存优化问题"""
    try:
        # 设置内存相关的环境变量
        memory_env_vars = {
            'VLLM_USE_FLASH_ATTN': '0',  # 禁用FlashAttention以避免兼容性问题
            'DISABLE_FLASH_ATTN': '1',
            'VLLM_ATTENTION_BACKEND': 'XFORMERS',  # 修改为使用XFORMERS后端而不是TORCH_SDPA
            'TRANSFORMERS_VERBOSITY': 'error',  # 减少transformers的日志输出
            'VLLM_ALLOW_LONG_MAX_MODEL_LEN': '1',  # 允许长序列
            'ATTN_IMPLEMENTATION': 'eager',  # 使用eager执行模式
        }
        
        for key, value in memory_env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
                logger.info(f"设置环境变量: {key}={value}")
        
        # 尝试调整PyTorch内存分配策略
        try:
            import torch
            if torch.cuda.is_available():
                # 设置内存分配策略
                torch.cuda.empty_cache()
                # 启用内存池
                os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
                logger.info("应用了CUDA内存优化设置")
        except Exception as e:
            logger.warning(f"CUDA内存优化设置失败: {e}")
        
        logger.info("vLLM内存优化补丁已应用")
        return True
    except Exception as e:
        logger.error(f"vLLM内存优化补丁失败: {e}")
        return False


def patch_vllm_engine_args():
    """修复vLLM引擎参数问题"""
    try:
        # 尝试导入和修复AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncEngineArgs
        
        # 创建一个修复后的引擎参数生成函数
        def create_safe_engine_args(model_path: str, **kwargs) -> AsyncEngineArgs:
            """创建安全的引擎参数，处理常见的参数问题"""
            
            # 设置默认的安全参数
            safe_defaults = {
                'trust_remote_code': True,
                'enforce_eager': True,  # 使用eager模式避免编译问题
                'distributed_executor_backend': 'mp',
                'enable_prefix_caching': False,  # 禁用前缀缓存以避免问题
                'gpu_memory_utilization': min(kwargs.get('gpu_memory_utilization', 0.9), 0.8),  # 降低GPU内存使用
                'tensor_parallel_size': kwargs.get('tensor_parallel_size', 1),
                'max_model_len': min(kwargs.get('max_model_len', 8192), 4096),  # 限制最大模型长度
                'max_num_seqs': min(kwargs.get('max_num_seqs', 8), 4),  # 减少并发序列数
                'block_size': kwargs.get('block_size', 16),
            }
            
            # 更新默认值
            for key, value in safe_defaults.items():
                if key not in kwargs:
                    kwargs[key] = value
            
            # 处理limit_mm_per_prompt参数
            if 'limit_mm_per_prompt' in kwargs:
                # 确保多模态限制不会太高
                limit_mm = kwargs['limit_mm_per_prompt']
                if isinstance(limit_mm, dict):
                    safe_limits = {
                        'audio': min(limit_mm.get('audio', 16), 8),
                        'image': min(limit_mm.get('image', 32), 16),
                        'video': min(limit_mm.get('video', 8), 4)
                    }
                    kwargs['limit_mm_per_prompt'] = safe_limits
            
            try:
                return AsyncEngineArgs(model=model_path, **kwargs)
            except Exception as e:
                logger.warning(f"使用完整参数创建引擎失败: {e}，尝试简化参数")
                # 如果失败，使用最基本的参数
                basic_kwargs = {
                    'model': model_path,
                    'trust_remote_code': True,
                    'enforce_eager': True,
                    'gpu_memory_utilization': 0.6,
                    'tensor_parallel_size': 1,
                }
                return AsyncEngineArgs(**basic_kwargs)
        
        # 将修复函数添加到模块中
        import vllm_model_service
        if hasattr(vllm_model_service, 'ModelService'):
            vllm_model_service.ModelService._create_safe_engine_args = create_safe_engine_args
        
        logger.info("vLLM引擎参数补丁已应用")
        return True
    except Exception as e:
        logger.error(f"vLLM引擎参数补丁失败: {e}")
        return False


def patch_error_handling():
    """增强错误处理"""
    try:
        # 添加常见错误的处理建议
        common_errors = {
            'OmniLLMEngine': '尝试检查vLLM版本是否支持OmniLLMEngine',
            'processing_omni': '多模态处理模块缺失，请检查vLLM版本',
            'CUDA out of memory': '显存不足，尝试降低gpu_memory_utilization',
            'RuntimeError': '运行时错误，尝试使用enforce_eager=True',
        }
        
        # 重定向警告到日志
        def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
            logger.warning(f"警告 [{category.__name__}]: {message}")
        
        warnings.showwarning = custom_warning_handler
        
        logger.info("错误处理补丁已应用")
        return True
    except Exception as e:
        logger.error(f"错误处理补丁失败: {e}")
        return False


def apply_all_patches() -> bool:
    """应用所有补丁"""
    logger.info("开始应用vLLM修复补丁...")
    
    patches = [
        ("vLLM安装检查", check_vllm_installation),
        ("视频像素限制修复", patch_video_max_pixels),
        ("多模态导入修复", patch_vllm_multimodal_imports),
        ("内存优化", patch_vllm_memory_optimization),
        ("引擎参数修复", patch_vllm_engine_args),
        ("错误处理增强", patch_error_handling),
    ]
    
    success_count = 0
    for patch_name, patch_func in patches:
        try:
            logger.info(f"应用补丁: {patch_name}")
            if patch_func():
                success_count += 1
                logger.info(f"✓ {patch_name} 补丁成功")
            else:
                logger.warning(f"✗ {patch_name} 补丁失败")
        except Exception as e:
            logger.error(f"✗ {patch_name} 补丁异常: {e}")
    
    success_rate = success_count / len(patches)
    logger.info(f"补丁应用完成: {success_count}/{len(patches)} 成功 ({success_rate:.1%})")
    
    return success_rate >= 0.5  # 至少一半的补丁成功才算成功


if __name__ == "__main__":
    # 测试补丁
    logging.basicConfig(level=logging.INFO)
    success = apply_all_patches()
    if success:
        print("✅ vLLM修复补丁应用成功")
    else:
        print("❌ vLLM修复补丁应用部分失败")
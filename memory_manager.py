#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显存和内存管理模块
支持动态配置哪些模块加载到显存，哪些保持在CPU
"""

import torch
import psutil
import gc
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import os

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """内存配置类"""
    # GPU内存配置
    gpu_memory_limit_gb: float = 4.0           # GPU内存限制(GB)
    gpu_memory_reserve_gb: float = 1.0         # GPU预留内存(GB)
    enable_gpu_memory_growth: bool = True      # 启用GPU内存增长
    
    # CPU内存配置
    cpu_memory_limit_gb: float = 8.0           # CPU内存限制(GB)
    enable_cpu_offload: bool = True            # 启用CPU卸载
    
    # 模块加载配置
    modules_on_gpu: List[str] = None           # 强制加载到GPU的模块
    modules_on_cpu: List[str] = None           # 强制加载到CPU的模块
    auto_manage_modules: bool = True           # 自动管理模块位置
    
    # 优化配置
    use_gradient_checkpointing: bool = True    # 梯度检查点
    use_half_precision: bool = True            # 半精度
    clear_cache_frequency: int = 5             # 缓存清理频率
    
    def __post_init__(self):
        if self.modules_on_gpu is None:
            self.modules_on_gpu = [
                "thinker.model.embed_tokens",
                "thinker.model.layers",
                "thinker.lm_head"
            ]
        
        if self.modules_on_cpu is None:
            self.modules_on_cpu = [
                "talker",
                "token2wav"
            ]

@dataclass
class ModuleMemoryInfo:
    """模块内存信息"""
    name: str
    parameters: int
    memory_mb: float
    device: str
    dtype: str
    requires_grad: bool
    is_active: bool = True

class MemoryManager:
    """内存管理器"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.module_info: Dict[str, ModuleMemoryInfo] = {}
        self.operation_count = 0
        self.setup_memory_monitoring()
    
    def setup_memory_monitoring(self):
        """设置内存监控"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 设置内存分配策略
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """获取系统内存信息"""
        info = {}
        
        # CPU内存
        memory = psutil.virtual_memory()
        info.update({
            'cpu_total_gb': memory.total / 1024**3,
            'cpu_used_gb': memory.used / 1024**3,
            'cpu_available_gb': memory.available / 1024**3,
            'cpu_percent': memory.percent
        })
        
        # GPU内存
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = props.total_memory / 1024**3
                
                info.update({
                    f'gpu_{i}_name': props.name,
                    f'gpu_{i}_total_gb': total,
                    f'gpu_{i}_allocated_gb': allocated,
                    f'gpu_{i}_reserved_gb': reserved,
                    f'gpu_{i}_free_gb': total - reserved,
                    f'gpu_{i}_utilization': (allocated / total) * 100 if total > 0 else 0
                })
        
        return info
    
    def analyze_model_memory_usage(self, model) -> Dict[str, ModuleMemoryInfo]:
        """分析模型内存使用情况"""
        module_info = {}
        
        def get_module_info(name: str, module: torch.nn.Module) -> ModuleMemoryInfo:
            # 计算参数数量
            params = sum(p.numel() for p in module.parameters())
            
            # 估算内存使用（字节）
            param_memory = 0
            device = "unknown"
            dtype = "unknown"
            requires_grad = False
            
            for param in module.parameters():
                param_memory += param.numel() * param.element_size()
                device = str(param.device)
                dtype = str(param.dtype)
                requires_grad = param.requires_grad
                break  # 假设模块内参数设备一致
            
            memory_mb = param_memory / 1024**2
            
            return ModuleMemoryInfo(
                name=name,
                parameters=params,
                memory_mb=memory_mb,
                device=device,
                dtype=dtype,
                requires_grad=requires_grad
            )
        
        # 分析主要模块
        try:
            if hasattr(model, 'thinker'):
                # Thinker模块
                if hasattr(model.thinker, 'model'):
                    module_info['thinker.model'] = get_module_info('thinker.model', model.thinker.model)
                
                if hasattr(model.thinker, 'lm_head'):
                    module_info['thinker.lm_head'] = get_module_info('thinker.lm_head', model.thinker.lm_head)
                
                if hasattr(model.thinker, 'visual'):
                    module_info['thinker.visual'] = get_module_info('thinker.visual', model.thinker.visual)
                
                if hasattr(model.thinker, 'audio_tower'):
                    module_info['thinker.audio_tower'] = get_module_info('thinker.audio_tower', model.thinker.audio_tower)
            
            # Talker模块
            if hasattr(model, 'talker'):
                module_info['talker'] = get_module_info('talker', model.talker)
            
            # Token2Wav模块
            if hasattr(model, 'token2wav'):
                module_info['token2wav'] = get_module_info('token2wav', model.token2wav)
            
        except Exception as e:
            logger.error(f"模型内存分析失败: {e}")
        
        self.module_info = module_info
        return module_info
    
    def apply_memory_optimization(self, model) -> Dict[str, Any]:
        """应用内存优化策略"""
        optimization_results = {
            'optimizations_applied': [],
            'memory_before': self.get_system_memory_info(),
            'module_moves': []
        }
        
        try:
            # 1. 强制GPU模块
            for module_name in self.config.modules_on_gpu:
                if self._move_module_to_device(model, module_name, 'cuda'):
                    optimization_results['module_moves'].append(f"{module_name} -> GPU")
                    optimization_results['optimizations_applied'].append(f"moved_{module_name}_to_gpu")
            
            # 2. 强制CPU模块
            for module_name in self.config.modules_on_cpu:
                if self._move_module_to_device(model, module_name, 'cpu'):
                    optimization_results['module_moves'].append(f"{module_name} -> CPU")
                    optimization_results['optimizations_applied'].append(f"moved_{module_name}_to_cpu")
            
            # 3. 自动管理模块（根据内存使用情况）
            if self.config.auto_manage_modules:
                auto_moves = self._auto_manage_modules(model)
                optimization_results['module_moves'].extend(auto_moves)
                optimization_results['optimizations_applied'].extend([f"auto_managed_{len(auto_moves)}_modules"])
            
            # 4. 应用梯度检查点
            if self.config.use_gradient_checkpointing:
                try:
                    model.gradient_checkpointing_enable()
                    optimization_results['optimizations_applied'].append("gradient_checkpointing")
                except:
                    logger.warning("梯度检查点设置失败")
            
            # 5. 清理内存
            self.clear_memory()
            optimization_results['optimizations_applied'].append("memory_cleanup")
            
            # 6. 记录优化后内存
            optimization_results['memory_after'] = self.get_system_memory_info()
            
            logger.info(f"内存优化完成: {len(optimization_results['optimizations_applied'])}项优化")
            
        except Exception as e:
            logger.error(f"内存优化失败: {e}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def _move_module_to_device(self, model, module_path: str, device: str) -> bool:
        """移动模块到指定设备"""
        try:
            # 解析模块路径
            module = model
            for attr in module_path.split('.'):
                if hasattr(module, attr):
                    module = getattr(module, attr)
                else:
                    logger.warning(f"模块路径不存在: {module_path}")
                    return False
            
            # 移动到指定设备
            module.to(device)
            logger.debug(f"模块 {module_path} 已移动到 {device}")
            return True
            
        except Exception as e:
            logger.error(f"模块移动失败 {module_path} -> {device}: {e}")
            return False
    
    def _auto_manage_modules(self, model) -> List[str]:
        """自动管理模块位置"""
        moves = []
        
        try:
            # 获取当前GPU内存使用
            if not torch.cuda.is_available():
                return moves
            
            current_memory = torch.cuda.memory_allocated() / 1024**3
            memory_limit = self.config.gpu_memory_limit_gb - self.config.gpu_memory_reserve_gb
            
            if current_memory > memory_limit:
                # 内存不足，将非关键模块移到CPU
                non_critical_modules = [
                    "thinker.visual",
                    "thinker.audio_tower",
                    "token2wav"
                ]
                
                for module_name in non_critical_modules:
                    if self._move_module_to_device(model, module_name, 'cpu'):
                        moves.append(f"{module_name} -> CPU (自动)")
                        
                        # 检查内存是否足够
                        new_memory = torch.cuda.memory_allocated() / 1024**3
                        if new_memory <= memory_limit:
                            break
            
        except Exception as e:
            logger.error(f"自动模块管理失败: {e}")
        
        return moves
    
    def clear_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def monitor_memory_during_inference(self, operation_name: str = "inference") -> Dict[str, float]:
        """推理期间监控内存"""
        self.operation_count += 1
        
        # 定期清理缓存
        if self.operation_count % self.config.clear_cache_frequency == 0:
            self.clear_memory()
            logger.debug(f"定期内存清理完成 (操作 #{self.operation_count})")
        
        # 获取内存信息
        memory_info = self.get_system_memory_info()
        
        # 检查内存警告
        if torch.cuda.is_available():
            gpu_util = memory_info.get('gpu_0_utilization', 0)
            if gpu_util > 90:
                logger.warning(f"GPU内存使用率过高: {gpu_util:.1f}%")
        
        cpu_percent = memory_info.get('cpu_percent', 0)
        if cpu_percent > 85:
            logger.warning(f"CPU内存使用率过高: {cpu_percent:.1f}%")
        
        return memory_info
    
    def get_optimization_recommendations(self) -> List[str]:
        """获取优化建议"""
        recommendations = []
        memory_info = self.get_system_memory_info()
        
        # GPU内存建议
        if torch.cuda.is_available():
            gpu_util = memory_info.get('gpu_0_utilization', 0)
            if gpu_util > 80:
                recommendations.append("GPU内存使用率过高，建议启用CPU卸载")
                recommendations.append("考虑减少batch_size或降低模型精度")
            elif gpu_util < 50:
                recommendations.append("GPU内存充足，可以提高模型精度或增加batch_size")
        
        # CPU内存建议
        cpu_percent = memory_info.get('cpu_percent', 0)
        if cpu_percent > 80:
            recommendations.append("CPU内存使用率过高，建议减少CPU卸载的模块")
        
        # 根据模块使用情况给出建议
        if self.module_info:
            total_gpu_memory = sum(
                info.memory_mb for info in self.module_info.values() 
                if 'cuda' in info.device
            )
            if total_gpu_memory > self.config.gpu_memory_limit_gb * 1024:
                recommendations.append("模型总内存超过限制，建议将更多模块移到CPU")
        
        return recommendations

class DynamicMemoryManager:
    """动态内存管理器"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_manager = MemoryManager(config)
        self.module_usage_history: Dict[str, int] = {}
        self.last_optimization_memory = 0
    
    def smart_module_placement(self, model) -> Dict[str, str]:
        """智能模块放置策略"""
        placement_plan = {}
        
        try:
            # 分析模型
            module_info = self.memory_manager.analyze_model_memory_usage(model)
            
            # 按内存使用排序
            sorted_modules = sorted(
                module_info.items(),
                key=lambda x: x[1].memory_mb,
                reverse=True
            )
            
            # 计算可用GPU内存
            memory_info = self.memory_manager.get_system_memory_info()
            available_gpu_memory = (
                self.config.gpu_memory_limit_gb - 
                self.config.gpu_memory_reserve_gb
            ) * 1024  # 转换为MB
            
            used_gpu_memory = 0
            
            # 优先级排序（核心模块优先GPU）
            core_modules = [
                "thinker.model.embed_tokens",
                "thinker.model.layers", 
                "thinker.lm_head"
            ]
            
            # 首先放置核心模块
            for module_name, info in sorted_modules:
                if any(core in module_name for core in core_modules):
                    if used_gpu_memory + info.memory_mb <= available_gpu_memory:
                        placement_plan[module_name] = "cuda"
                        used_gpu_memory += info.memory_mb
                    else:
                        placement_plan[module_name] = "cpu"
                        logger.warning(f"核心模块 {module_name} 被放置到CPU（内存不足）")
            
            # 然后放置其他模块
            for module_name, info in sorted_modules:
                if module_name not in placement_plan:
                    if used_gpu_memory + info.memory_mb <= available_gpu_memory:
                        placement_plan[module_name] = "cuda"
                        used_gpu_memory += info.memory_mb
                    else:
                        placement_plan[module_name] = "cpu"
            
            logger.info(f"智能放置计划: GPU {used_gpu_memory:.1f}MB / {available_gpu_memory:.1f}MB")
            
        except Exception as e:
            logger.error(f"智能模块放置失败: {e}")
        
        return placement_plan
    
    def adaptive_optimization(self, model, current_memory_usage: float) -> bool:
        """自适应优化"""
        try:
            # 如果内存使用超过阈值，触发优化
            memory_threshold = self.config.gpu_memory_limit_gb * 0.9
            
            if current_memory_usage > memory_threshold:
                logger.info(f"内存使用过高 ({current_memory_usage:.2f}GB)，触发自适应优化")
                
                # 找到最少使用的非核心模块移到CPU
                candidates = [
                    "thinker.visual",
                    "thinker.audio_tower",
                    "token2wav"
                ]
                
                for candidate in candidates:
                    if self.memory_manager._move_module_to_device(model, candidate, 'cpu'):
                        logger.info(f"自适应优化: 移动 {candidate} 到CPU")
                        
                        # 清理内存并检查
                        self.memory_manager.clear_memory()
                        new_usage = torch.cuda.memory_allocated() / 1024**3
                        
                        if new_usage <= memory_threshold:
                            logger.info(f"自适应优化成功: {current_memory_usage:.2f}GB -> {new_usage:.2f}GB")
                            return True
                
                logger.warning("自适应优化无法充分释放内存")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"自适应优化失败: {e}")
            return False

class ConfigurableMemoryLoader:
    """可配置的内存加载器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.memory_manager = MemoryManager(self.config)
        self.dynamic_manager = DynamicMemoryManager(self.config)
    
    def load_config(self, config_path: Optional[str]) -> MemoryConfig:
        """加载内存配置"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                return MemoryConfig(**config_dict)
            except Exception as e:
                logger.error(f"配置加载失败: {e}")
        
        # 返回默认配置
        return MemoryConfig()
    
    def save_config(self, config_path: str):
        """保存内存配置"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存: {config_path}")
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
    
    def load_model_with_memory_config(self, model_class, model_path: str, **model_kwargs):
        """使用内存配置加载模型"""
        try:
            logger.info("开始加载模型并应用内存配置")
            
            # 获取智能设备映射
            device_map = self.get_intelligent_device_map()
            
            # 合并配置
            final_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch.float16 if self.config.use_half_precision else torch.float32,
                **model_kwargs
            }
            
            # 加载模型
            model = model_class.from_pretrained(model_path, **final_kwargs)
            
            # 应用内存优化
            optimization_results = self.memory_manager.apply_memory_optimization(model)
            
            logger.info("模型加载和内存优化完成")
            self._print_memory_summary()
            
            return model, optimization_results
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def get_intelligent_device_map(self) -> Dict[str, str]:
        """获取智能设备映射"""
        device_map = {}
        
        # 检查GPU可用性和内存
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，所有模块将加载到CPU")
            return "cpu"
        
        memory_info = self.memory_manager.get_system_memory_info()
        available_gpu_memory = memory_info.get('gpu_0_free_gb', 0)
        
        if available_gpu_memory < 2.0:
            logger.warning("GPU内存不足，启用激进的CPU卸载")
            device_map.update({
                "thinker.model": "cuda",
                "thinker.lm_head": "cuda",
                "thinker.visual": "cpu",
                "thinker.audio_tower": "cpu",
                "talker": "cpu",
                "token2wav": "cpu"
            })
        elif available_gpu_memory < 4.0:
            logger.info("GPU内存有限，启用适度的CPU卸载")
            device_map.update({
                "thinker.model": "cuda",
                "thinker.lm_head": "cuda", 
                "thinker.visual": "cuda",
                "thinker.audio_tower": "cpu",
                "talker": "cpu",
                "token2wav": "cpu"
            })
        else:
            logger.info("GPU内存充足，主要模块加载到GPU")
            device_map.update({
                "thinker.model": "cuda",
                "thinker.lm_head": "cuda",
                "thinker.visual": "cuda", 
                "thinker.audio_tower": "cuda",
                "talker": "cpu",      # talker仍保持在CPU（根据用户需求）
                "token2wav": "cpu"
            })
        
        return device_map
    
    def _print_memory_summary(self):
        """打印内存摘要"""
        try:
            memory_info = self.memory_manager.get_system_memory_info()
            
            print("\n" + "="*60)
            print("📊 内存使用摘要")
            print("="*60)
            
            # CPU内存
            print(f"💻 CPU内存: {memory_info['cpu_used_gb']:.2f}GB / {memory_info['cpu_total_gb']:.2f}GB ({memory_info['cpu_percent']:.1f}%)")
            
            # GPU内存
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_name = memory_info.get(f'gpu_{i}_name', f'GPU {i}')
                    gpu_used = memory_info.get(f'gpu_{i}_allocated_gb', 0)
                    gpu_total = memory_info.get(f'gpu_{i}_total_gb', 0)
                    gpu_util = memory_info.get(f'gpu_{i}_utilization', 0)
                    
                    print(f"🎮 {gpu_name}: {gpu_used:.2f}GB / {gpu_total:.2f}GB ({gpu_util:.1f}%)")
            
            # 优化建议
            recommendations = self.memory_manager.get_optimization_recommendations()
            if recommendations:
                print(f"\n💡 优化建议:")
                for rec in recommendations:
                    print(f"  - {rec}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"内存摘要打印失败: {e}")

# 预设内存配置
class MemoryPresets:
    """内存配置预设"""
    
    @staticmethod
    def get_preset(name: str) -> MemoryConfig:
        """获取预设配置"""
        presets = {
            'ultra_low_vram': MemoryConfig(
                gpu_memory_limit_gb=2.0,
                gpu_memory_reserve_gb=0.5,
                modules_on_gpu=["thinker.model.layers"],
                modules_on_cpu=[
                    "thinker.visual",
                    "thinker.audio_tower", 
                    "talker",
                    "token2wav"
                ],
                auto_manage_modules=True,
                use_gradient_checkpointing=True,
                use_half_precision=True
            ),
            
            'low_vram': MemoryConfig(
                gpu_memory_limit_gb=4.0,
                gpu_memory_reserve_gb=1.0,
                modules_on_gpu=[
                    "thinker.model",
                    "thinker.lm_head"
                ],
                modules_on_cpu=[
                    "talker",
                    "token2wav"
                ],
                auto_manage_modules=True,
                use_gradient_checkpointing=True,
                use_half_precision=True
            ),
            
            'balanced': MemoryConfig(
                gpu_memory_limit_gb=6.0,
                gpu_memory_reserve_gb=1.5,
                modules_on_gpu=[
                    "thinker.model",
                    "thinker.lm_head",
                    "thinker.visual"
                ],
                modules_on_cpu=[
                    "talker",
                    "token2wav"
                ],
                auto_manage_modules=True,
                use_gradient_checkpointing=False,
                use_half_precision=True
            ),
            
            'high_vram': MemoryConfig(
                gpu_memory_limit_gb=12.0,
                gpu_memory_reserve_gb=2.0,
                modules_on_gpu=[
                    "thinker",
                    "thinker.visual",
                    "thinker.audio_tower"
                ],
                modules_on_cpu=[],
                auto_manage_modules=False,
                use_gradient_checkpointing=False,
                use_half_precision=False
            )
        }
        
        if name not in presets:
            raise ValueError(f"未知预设: {name}. 可用预设: {list(presets.keys())}")
        
        return presets[name]
    
    @staticmethod
    def list_presets() -> List[str]:
        """列出所有预设"""
        return ['ultra_low_vram', 'low_vram', 'balanced', 'high_vram']
    
    @staticmethod
    def detect_optimal_preset() -> str:
        """检测最优预设"""
        try:
            if not torch.cuda.is_available():
                return 'ultra_low_vram'
            
            # 检测GPU内存
            props = torch.cuda.get_device_properties(0)
            total_memory_gb = props.total_memory / 1024**3
            
            if total_memory_gb < 4:
                return 'ultra_low_vram'
            elif total_memory_gb < 6:
                return 'low_vram'
            elif total_memory_gb < 10:
                return 'balanced'
            else:
                return 'high_vram'
                
        except Exception:
            return 'ultra_low_vram'

def create_memory_config_file(preset_name: str, output_path: str):
    """创建内存配置文件"""
    try:
        config = MemoryPresets.get_preset(preset_name)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)
        
        logger.info(f"内存配置文件已创建: {output_path}")
        
    except Exception as e:
        logger.error(f"配置文件创建失败: {e}")

def auto_configure_memory() -> MemoryConfig:
    """自动配置内存"""
    optimal_preset = MemoryPresets.detect_optimal_preset()
    logger.info(f"检测到最优预设: {optimal_preset}")
    return MemoryPresets.get_preset(optimal_preset)
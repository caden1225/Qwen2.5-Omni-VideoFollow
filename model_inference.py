#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理API模块
专门用于Qwen2.5-Omni模型的推理，支持显存配置和模块选择性加载
"""

import os
import sys
import torch
import gc
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
import json

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "qwen-omni-utils" / "src"))
sys.path.insert(0, str(project_root / "low-VRAM-mode"))

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """模型配置类"""
    model_path: str = "/home/caden/workplace/models/Qwen2.5-Omni-3B"
    load_thinker_only: bool = True         # 只加载thinker部分
    load_talker: bool = False              # 是否加载talker（语音合成）
    use_half_precision: bool = True        # 使用半精度
    device_map: Optional[Dict[str, str]] = None  # 设备映射
    torch_dtype: torch.dtype = torch.float16
    attn_implementation: str = "eager"     # 注意力实现方式
    trust_remote_code: bool = True
    
    # 生成参数
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    pad_token_id: Optional[int] = None

@dataclass 
class VRAMConfig:
    """显存配置类"""
    enable_cpu_offload: bool = True        # 启用CPU卸载
    offload_modules: List[str] = None      # 要卸载到CPU的模块
    enable_gradient_checkpointing: bool = True  # 梯度检查点
    use_8bit: bool = False                 # 使用8bit量化
    use_4bit: bool = False                 # 使用4bit量化
    max_memory_per_gpu: str = "auto"       # 每个GPU最大内存
    
    def __post_init__(self):
        if self.offload_modules is None:
            self.offload_modules = [
                "thinker.visual",
                "thinker.audio_tower", 
                "talker",
                "token2wav"
            ]

class QwenOmniInference:
    """Qwen Omni推理引擎"""
    
    def __init__(self, model_config: ModelConfig, vram_config: VRAMConfig):
        self.model_config = model_config
        self.vram_config = vram_config
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.setup_environment()
        self.load_model()
    
    def setup_environment(self):
        """设置环境变量"""
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 设置CUDA内存策略
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    
    def get_optimized_device_map(self) -> Dict[str, str]:
        """获取优化的设备映射"""
        if self.model_config.device_map:
            return self.model_config.device_map
        
        # 根据配置生成设备映射
        device_map = {}
        
        if self.model_config.load_thinker_only:
            # 只加载thinker相关模块
            device_map.update({
                "thinker.model": "cuda",
                "thinker.lm_head": "cuda",
            })
            
            # 根据VRAM配置决定visual和audio的位置
            if self.vram_config.enable_cpu_offload:
                device_map.update({
                    "thinker.visual": "cpu",
                    "thinker.audio_tower": "cpu",
                })
            else:
                device_map.update({
                    "thinker.visual": "cuda",
                    "thinker.audio_tower": "cuda",
                })
            
            # talker相关模块
            if not self.model_config.load_talker:
                device_map.update({
                    "talker": "cpu",
                    "token2wav": "cpu",
                })
            else:
                device_map.update({
                    "talker": "cuda",
                    "token2wav": "cuda",
                })
        else:
            # 加载完整模型
            device_map = "auto"
        
        logger.info(f"设备映射: {device_map}")
        return device_map
    
    def load_model(self):
        """加载模型"""
        try:
            logger.info(f"开始加载模型: {self.model_config.model_path}")
            
            # 检查模型路径
            if not os.path.exists(self.model_config.model_path):
                raise ValueError(f"模型路径不存在: {self.model_config.model_path}")
            
            # 导入模型类
            try:
                from low_VRAM_mode.modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration
                from transformers import Qwen2_5OmniProcessor
                logger.info("使用低显存模式模型")
            except ImportError:
                try:
                    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
                    logger.info("使用标准模型")
                except ImportError:
                    logger.error("无法导入Qwen2.5-Omni模型，请检查transformers版本")
                    raise
            
            # 获取设备映射
            device_map = self.get_optimized_device_map()
            
            # 加载模型
            model_kwargs = {
                "torch_dtype": self.model_config.torch_dtype,
                "attn_implementation": self.model_config.attn_implementation,
                "trust_remote_code": self.model_config.trust_remote_code,
            }
            
            if isinstance(device_map, dict):
                model_kwargs["device_map"] = device_map
            
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_config.model_path,
                **model_kwargs
            )
            
            # 加载处理器
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_config.model_path)
            
            # 设置pad_token_id
            if self.model_config.pad_token_id is None:
                self.model_config.pad_token_id = self.processor.tokenizer.eos_token_id
            
            # 配置模型
            if self.vram_config.enable_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            logger.info("模型加载成功")
            self._print_model_info()
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _print_model_info(self):
        """打印模型信息"""
        try:
            # 计算模型参数数量
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
            
            # 显存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"显存使用: {allocated:.2f}GB 已分配, {cached:.2f}GB 已缓存")
            
        except Exception as e:
            logger.warning(f"获取模型信息失败: {e}")
    
    def text_inference(self, messages: List[Dict], **generation_kwargs) -> str:
        """纯文本推理"""
        try:
            # 清理内存
            self.clear_cache()
            
            # 应用聊天模板
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 处理输入
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # 合并生成参数
            gen_kwargs = {
                "max_new_tokens": self.model_config.max_new_tokens,
                "temperature": self.model_config.temperature,
                "top_p": self.model_config.top_p,
                "do_sample": self.model_config.do_sample,
                "pad_token_id": self.model_config.pad_token_id,
                **generation_kwargs
            }
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # 解码输出
            response = self.processor.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # 清理内存
            del inputs, outputs
            self.clear_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"文本推理失败: {e}")
            self.clear_cache()
            raise
    
    def multimodal_inference(self, messages: List[Dict], 
                           use_audio_in_video: bool = True,
                           return_audio: bool = False,
                           **generation_kwargs) -> Union[str, Tuple[str, torch.Tensor]]:
        """多模态推理"""
        try:
            # 清理内存
            self.clear_cache()
            
            # 应用聊天模板
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 处理多模态信息
            try:
                from qwen_omni_utils import process_mm_info
                audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
            except ImportError:
                logger.warning("qwen_omni_utils不可用，使用基础处理")
                audios, images, videos = None, None, None
            
            # 处理输入
            inputs = self.processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True
            )
            inputs = inputs.to(self.device)
            
            # 清理中间变量
            del audios, images, videos
            self.clear_cache()
            
            # 合并生成参数
            gen_kwargs = {
                "max_new_tokens": self.model_config.max_new_tokens,
                "temperature": self.model_config.temperature,
                "top_p": self.model_config.top_p,
                "do_sample": self.model_config.do_sample,
                "pad_token_id": self.model_config.pad_token_id,
                "use_audio_in_video": use_audio_in_video,
                "return_audio": return_audio,
                **generation_kwargs
            }
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # 处理输出
            if return_audio and len(outputs) > 2:
                # 多模态输出（文本+音频）
                text_output = self.processor.batch_decode(
                    outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                audio_output = outputs[2]
                
                # 清理内存
                del inputs, outputs
                self.clear_cache()
                
                return text_output, audio_output
            else:
                # 纯文本输出
                response = self.processor.batch_decode(
                    outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                # 清理内存
                del inputs, outputs
                self.clear_cache()
                
                return response
            
        except Exception as e:
            logger.error(f"多模态推理失败: {e}")
            self.clear_cache()
            raise
    
    def video_inference(self, video_path: str, 
                       prompt: str = None,
                       system_prompt: str = None,
                       extract_audio: bool = True,
                       extract_last_frame: bool = True,
                       **generation_kwargs) -> Union[str, Tuple[str, torch.Tensor]]:
        """
        视频推理
        
        Args:
            video_path: 视频文件路径
            prompt: 用户提示
            system_prompt: 系统提示
            extract_audio: 是否提取音频
            extract_last_frame: 是否只提取最后一帧
            
        Returns:
            推理结果（文本或文本+音频）
        """
        try:
            from video_utils import create_conversation_with_video
            
            # 创建对话格式
            messages = []
            
            # 添加系统提示
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            
            # 创建视频对话
            video_messages = create_conversation_with_video(
                video_path, prompt, extract_audio, extract_last_frame
            )
            messages.extend(video_messages)
            
            # 执行推理
            return self.multimodal_inference(
                messages, 
                use_audio_in_video=extract_audio,
                return_audio=False,  # 暂时不返回音频
                **generation_kwargs
            )
            
        except Exception as e:
            logger.error(f"视频推理失败: {e}")
            raise
    
    def clear_cache(self):
        """清理缓存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_info(self) -> Dict[str, float]:
        """获取内存使用信息"""
        info = {}
        
        if torch.cuda.is_available():
            info.update({
                'gpu_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'gpu_cached_gb': torch.cuda.memory_reserved() / 1024**3,
                'gpu_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        
        return info
    
    def unload_unused_modules(self):
        """卸载未使用的模块到CPU"""
        if not self.model:
            return
        
        try:
            for module_name in self.vram_config.offload_modules:
                if hasattr(self.model, module_name.split('.')[0]):
                    module = self.model
                    for attr in module_name.split('.'):
                        if hasattr(module, attr):
                            module = getattr(module, attr)
                        else:
                            break
                    else:
                        # 移动到CPU
                        module.to('cpu')
                        logger.debug(f"模块 {module_name} 已移动到CPU")
            
            self.clear_cache()
            logger.info("未使用模块已卸载到CPU")
            
        except Exception as e:
            logger.warning(f"模块卸载失败: {e}")

class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.inference_engine = None
        self.current_config = None
    
    def load_model_with_config(self, model_config: ModelConfig, vram_config: VRAMConfig) -> bool:
        """使用指定配置加载模型"""
        try:
            # 如果已有模型，先清理
            if self.inference_engine:
                self.cleanup()
            
            # 创建推理引擎
            self.inference_engine = QwenOmniInference(model_config, vram_config)
            self.current_config = (model_config, vram_config)
            
            logger.info("模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def inference(self, input_type: str, **kwargs) -> Any:
        """执行推理"""
        if not self.inference_engine:
            raise ValueError("模型未加载")
        
        if input_type == "text":
            return self.inference_engine.text_inference(**kwargs)
        elif input_type == "multimodal":
            return self.inference_engine.multimodal_inference(**kwargs)
        elif input_type == "video":
            return self.inference_engine.video_inference(**kwargs)
        else:
            raise ValueError(f"不支持的推理类型: {input_type}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        if self.inference_engine:
            return self.inference_engine.get_memory_info()
        return {}
    
    def cleanup(self):
        """清理资源"""
        if self.inference_engine:
            self.inference_engine.clear_cache()
            del self.inference_engine.model
            del self.inference_engine.processor
            self.inference_engine = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("模型资源已清理")

# 预设配置
class InferencePresets:
    """推理预设配置"""
    
    @staticmethod
    def get_model_preset(name: str) -> Tuple[ModelConfig, VRAMConfig]:
        """获取模型预设配置"""
        presets = {
            'ultra_low_vram': (
                ModelConfig(
                    load_thinker_only=True,
                    load_talker=False,
                    use_half_precision=True,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=False  # 贪婪解码减少内存
                ),
                VRAMConfig(
                    enable_cpu_offload=True,
                    enable_gradient_checkpointing=True,
                    offload_modules=[
                        "thinker.visual",
                        "thinker.audio_tower",
                        "talker",
                        "token2wav"
                    ]
                )
            ),
            
            'low_vram': (
                ModelConfig(
                    load_thinker_only=True,
                    load_talker=False,
                    use_half_precision=True,
                    max_new_tokens=256,
                    temperature=0.7
                ),
                VRAMConfig(
                    enable_cpu_offload=True,
                    enable_gradient_checkpointing=True,
                    offload_modules=[
                        "thinker.visual",
                        "thinker.audio_tower",
                        "token2wav"
                    ]
                )
            ),
            
            'balanced': (
                ModelConfig(
                    load_thinker_only=False,
                    load_talker=False,
                    use_half_precision=True,
                    max_new_tokens=512,
                    temperature=0.7
                ),
                VRAMConfig(
                    enable_cpu_offload=False,
                    enable_gradient_checkpointing=True,
                    offload_modules=["talker", "token2wav"]
                )
            ),
            
            'full_model': (
                ModelConfig(
                    load_thinker_only=False,
                    load_talker=True,
                    use_half_precision=False,
                    max_new_tokens=1024,
                    temperature=0.7
                ),
                VRAMConfig(
                    enable_cpu_offload=False,
                    enable_gradient_checkpointing=False,
                    offload_modules=[]
                )
            )
        }
        
        if name not in presets:
            raise ValueError(f"未知预设: {name}. 可用预设: {list(presets.keys())}")
        
        return presets[name]
    
    @staticmethod
    def list_presets() -> List[str]:
        """列出所有预设"""
        return ['ultra_low_vram', 'low_vram', 'balanced', 'full_model']
    
    @staticmethod
    def get_preset_info() -> Dict[str, Dict[str, Any]]:
        """获取预设信息"""
        return {
            'ultra_low_vram': {
                'description': '超低显存模式',
                'vram_usage': '< 3GB',
                'features': ['仅thinker', 'CPU卸载', '半精度'],
                'performance': '低'
            },
            'low_vram': {
                'description': '低显存模式',
                'vram_usage': '< 4GB', 
                'features': ['仅thinker', '部分CPU卸载', '半精度'],
                'performance': '中低'
            },
            'balanced': {
                'description': '平衡模式',
                'vram_usage': '< 6GB',
                'features': ['完整thinker', '梯度检查点', '半精度'], 
                'performance': '中'
            },
            'full_model': {
                'description': '完整模型',
                'vram_usage': '> 8GB',
                'features': ['完整模型', '包含talker', '全精度'],
                'performance': '高'
            }
        }

# 全局模型管理器
model_manager = ModelManager()

def load_model(preset_name: str = "low_vram", model_path: str = None) -> bool:
    """加载模型的便捷函数"""
    try:
        model_config, vram_config = InferencePresets.get_model_preset(preset_name)
        
        if model_path:
            model_config.model_path = model_path
        
        return model_manager.load_model_with_config(model_config, vram_config)
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return False

def inference(input_type: str, **kwargs) -> Any:
    """执行推理的便捷函数"""
    return model_manager.inference(input_type, **kwargs)

def get_memory_info() -> Dict[str, float]:
    """获取内存信息的便捷函数"""
    return model_manager.get_memory_usage()

def cleanup_model():
    """清理模型的便捷函数"""
    model_manager.cleanup()
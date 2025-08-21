#!/usr/bin/env python3
"""
Qwen2.5-Omni vLLM模型服务
使用vLLM启动模型并提供API服务，支持多模态输入和输出
"""

import os
import time
import logging
import json
import base64
import uuid
import queue
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

import torch
import numpy as np
import librosa
import cv2
from PIL import Image
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# 导入qwen-omni-utils
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义请求和响应模型
class MultimodalRequest(BaseModel):
    text: Optional[str] = Field(None, description="文本输入")
    system_prompt: str = Field("You are a helpful AI assistant.", description="系统提示")
    max_tokens: int = Field(512, description="最大生成token数")
    extract_video_audio: bool = Field(False, description="是否提取视频音轨")
    extract_video_frame: bool = Field(False, description="是否提取视频最后一帧")
    using_mm_info_audio: bool = Field(False, description="是否使用mm_info提取音频")
    enable_audio_output: bool = Field(False, description="是否启用语音输出")
    enable_streaming: bool = Field(False, description="是否启用流式输出")
    
class MultimodalResponse(BaseModel):
    status: str = Field(..., description="处理状态")
    response_text: str = Field(..., description="生成的文本回答")
    extracted_audio: Optional[str] = Field(None, description="提取的音频文件路径")
    extracted_frame: Optional[str] = Field(None, description="提取的图像文件路径")
    generated_audio: Optional[str] = Field(None, description="生成的音频文件路径")
    processing_time: float = Field(..., description="处理时间")
    peak_memory: float = Field(..., description="峰值显存使用")
    error: Optional[str] = Field(None, description="错误信息")

class ModelService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.getenv("MODEL_PATH", "/home/caden/models/Qwen2.5-Omni-3B")
        self.temp_dir = Path("temp_files")
        self.temp_dir.mkdir(exist_ok=True)
        
        # vLLM相关配置
        self.use_vllm = os.getenv("USE_VLLM", "true").lower() == "true"
        self.vllm_model_path = os.getenv("VLLM_MODEL_PATH", self.model_path)
        self.vllm_tensor_parallel_size = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
        self.vllm_max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
        self.vllm_gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
        
        # 兼容性和回退机制标志
        self.vllm_available = False
        self.fallback_mode = False
        self.compatibility_issues = []
        
    def _print_memory_usage(self, stage: str):
        """打印不同阶段的显存占用统计"""
        if not torch.cuda.is_available():
            return
            
        print(f"\n🔍 {stage} - 显存占用统计:")
        print(f"   当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   显存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"   显存峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        # 获取GPU信息
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            free = total_memory - cached
            
            print(f"   GPU {i} ({props.name}):")
            print(f"     总显存: {total_memory:.2f} GB")
            print(f"     已分配: {allocated:.2f} GB")
            print(f"     已缓存: {cached:.2f} GB")
            print(f"     可用显存: {free:.2f} GB")
        print()
        
    def _check_vllm_compatibility(self):
        """检查vLLM兼容性"""
        try:
            import vllm
            vllm_version = getattr(vllm, '__version__', 'unknown')
            print(f"🔍 检测到vLLM版本: {vllm_version}")
            
            # 检查关键模块
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
                    __import__(module_name)
                except ImportError:
                    missing_modules.append(module_name)
            
            if missing_modules:
                print(f"⚠️ 缺失模块: {', '.join(missing_modules)}")
                return False
                
            return True
        except Exception as e:
            print(f"⚠️ vLLM兼容性检查失败: {e}")
            return False

    def _create_optimized_engine_args(self):
        """创建优化的引擎参数"""
        try:
            from vllm.engine.async_llm_engine import AsyncEngineArgs
            
            # 保守的参数配置，优先稳定性
            config = {
                'model': self.vllm_model_path,
                'trust_remote_code': True,
                'enforce_eager': True,  # 强制使用eager模式避免编译问题
                'distributed_executor_backend': 'mp',
                'enable_prefix_caching': False,  # 禁用前缀缓存避免兼容性问题
                'gpu_memory_utilization': min(self.vllm_gpu_memory_utilization, 0.75),  # 保守的内存使用
                'tensor_parallel_size': self.vllm_tensor_parallel_size,
                'max_model_len': min(self.vllm_max_model_len, 4096),  # 限制序列长度
                'max_num_seqs': min(8, 4),  # 减少并发数
                'block_size': 16,
            }
            
            # 动态调整多模态限制
            gpu_memory_gb = 0
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"🖥️ GPU显存: {gpu_memory_gb:.1f} GB")
            
            # 根据显存动态调整多模态限制
            if gpu_memory_gb >= 24:  # 24GB以上显存
                mm_limits = {'audio': 16, 'image': 32, 'video': 8}
            elif gpu_memory_gb >= 16:  # 16-24GB显存
                mm_limits = {'audio': 12, 'image': 24, 'video': 6}
            elif gpu_memory_gb >= 8:   # 8-16GB显存
                mm_limits = {'audio': 8, 'image': 16, 'video': 4}
            else:  # 8GB以下显存
                mm_limits = {'audio': 4, 'image': 8, 'video': 2}
                config['max_model_len'] = min(config['max_model_len'], 2048)
                config['gpu_memory_utilization'] = min(config['gpu_memory_utilization'], 0.6)
            
            config['limit_mm_per_prompt'] = mm_limits
            print(f"🎛️ 多模态限制: {mm_limits}")
            
            return AsyncEngineArgs(**config)
            
        except Exception as e:
            print(f"❌ 创建引擎参数失败: {e}")
            # 返回最基础的配置
            try:
                from vllm.engine.async_llm_engine import AsyncEngineArgs
                basic_config = {
                    'model': self.vllm_model_path,
                    'trust_remote_code': True,
                    'enforce_eager': True,
                    'gpu_memory_utilization': 0.6,
                    'tensor_parallel_size': 1,
                }
                print("🔄 使用基础配置重试")
                return AsyncEngineArgs(**basic_config)
            except Exception as e2:
                print(f"❌ 基础配置也失败: {e2}")
                return None

    def _check_model_files(self, model_path: str) -> bool:
        """检查模型文件完整性"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                print(f"❌ 模型路径不存在: {model_path}")
                return False
            
            # 检查关键文件
            required_files = [
                'config.json',
                'tokenizer.json',
                'tokenizer_config.json'
            ]
            
            missing_files = []
            for file_name in required_files:
                if not (model_path / file_name).exists():
                    missing_files.append(file_name)
            
            if missing_files:
                print(f"⚠️ 缺失关键文件: {', '.join(missing_files)}")
                self.compatibility_issues.append(f"缺失文件: {', '.join(missing_files)}")
                return False
            
            print("✅ 模型文件完整性检查通过")
            return True
        except Exception as e:
            print(f"❌ 模型文件检查失败: {e}")
            self.compatibility_issues.append(f"文件检查错误: {e}")
            return False

    def _try_fallback_mode(self):
        """尝试回退模式（使用标准transformers）"""
        try:
            print("🔄 启动回退模式，使用标准transformers加载...")
            from transformers import AutoModel, AutoTokenizer, AutoProcessor
            
            # 加载基础模型和tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # 尝试加载处理器
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                print("✅ 回退模式处理器加载成功")
            except Exception as e:
                print(f"⚠️ 回退模式处理器加载失败: {e}")
                # 使用基础的Qwen2处理器
                try:
                    from transformers import Qwen2Processor
                    self.processor = Qwen2Processor.from_pretrained(self.model_path)
                    print("✅ 使用基础Qwen2处理器")
                except Exception as e2:
                    print(f"❌ 基础处理器也失败: {e2}")
                    return False
            
            self.fallback_mode = True
            print("✅ 回退模式启动成功")
            return True
            
        except Exception as e:
            print(f"❌ 回退模式失败: {e}")
            self.compatibility_issues.append(f"回退模式错误: {e}")
            return False

    def _provide_compatibility_report(self):
        """提供兼容性报告"""
        print("\n" + "="*60)
        print("🔍 兼容性诊断报告")
        print("="*60)
        
        diagnostic = self._diagnose_system_state()
        for key, value in diagnostic.items():
            print(f"  {key}: {value}")
        
        if self.compatibility_issues:
            print("\n⚠️ 发现的问题:")
            for i, issue in enumerate(self.compatibility_issues, 1):
                print(f"  {i}. {issue}")
        
        print(f"\n🎯 当前状态:")
        print(f"  - vLLM可用: {self.vllm_available}")
        print(f"  - 回退模式: {self.fallback_mode}")
        print(f"  - 模型已加载: {hasattr(self, 'vllm_model') or self.fallback_mode}")
        
        if not self.vllm_available and not self.fallback_mode:
            print("\n💡 建议:")
            print("  1. 检查vLLM安装: pip install vllm")
            print("  2. 更新transformers: pip install -U transformers")
            print("  3. 检查CUDA版本兼容性")
            print("  4. 确认模型文件完整性")
            print("  5. 尝试降低GPU内存使用率")
        
        print("="*60)

    def load_model(self):
        """加载模型和处理器"""
        try:
            # 检查模型文件完整性
            if not self._check_model_files(self.vllm_model_path):
                print("❌ 模型文件检查失败")
                self._provide_compatibility_report()
                return False
            
            if self.use_vllm:
                print(f"🚀 正在使用vLLM加载模型: {self.vllm_model_path}")
                
                # 设置视频处理相关的环境变量
                os.environ.setdefault('VIDEO_MAX_PIXELS', str(32000 * 28 * 28))
                print(f"🔧 设置VIDEO_MAX_PIXELS: {os.environ.get('VIDEO_MAX_PIXELS')}")
                
                # 应用修复补丁
                try:
                    from vllm_fix_patch import apply_all_patches
                    if apply_all_patches():
                        print("✅ vLLM修复补丁应用成功")
                    else:
                        print("⚠️ vLLM修复补丁应用部分失败，继续尝试加载")
                except ImportError:
                    print("⚠️ 修复补丁文件未找到，继续尝试加载")
                except Exception as e:
                    print(f"⚠️ 修复补丁应用失败: {e}，继续尝试加载")
                
                # 导入vLLM相关模块
                try:
                    from vllm.engine.omni_llm_engine import OmniLLMEngine
                    from vllm.engine.async_llm_engine import AsyncEngineArgs
                    from vllm.sampling_params import SamplingParams
                    from vllm.inputs import TextPrompt
                    from vllm.multimodal.processing_omni import fetch_image, fetch_video
                except ImportError as e:
                    print(f"❌ vLLM导入失败: {e}")
                    print("请检查vLLM安装: pip install vllm")
                    return False
                
                # 创建优化的引擎参数
                print("🔧 创建优化的引擎配置...")
                thinker_engine_args = self._create_optimized_engine_args()
                if thinker_engine_args is None:
                    print("❌ 无法创建引擎参数")
                    return False
                
                # 尝试初始化OmniLLMEngine，使用多种回退策略
                print("🚀 初始化OmniLLMEngine...")
                try:
                    self.vllm_model = OmniLLMEngine(
                        thinker_engine_args,
                        thinker_visible_devices=[0],
                    )
                    print("✅ OmniLLMEngine初始化成功")
                except Exception as e:
                    print(f"❌ OmniLLMEngine初始化失败: {e}")
                    print("🔄 尝试使用简化配置...")
                    
                    # 尝试简化配置
                    try:
                        simple_args = AsyncEngineArgs(
                            model=self.vllm_model_path,
                            trust_remote_code=True,
                            enforce_eager=True,
                            gpu_memory_utilization=0.5,
                            tensor_parallel_size=1,
                        )
                        self.vllm_model = OmniLLMEngine(
                            simple_args,
                            thinker_visible_devices=[0],
                        )
                        print("✅ 使用简化配置初始化成功")
                    except Exception as e2:
                        print(f"❌ 简化配置也失败: {e2}")
                
                print("📦 vLLM模型加载完成")
                self.vllm_available = True
                
                # 统计显存占用
                if torch.cuda.is_available():
                    self._print_memory_usage("vLLM模型加载完成后")
                
                # 加载处理器（用于多模态输入处理）
                try:
                    self.processor = Qwen2_5OmniProcessor.from_pretrained(self.vllm_model_path)
                    print("✅ 处理器加载完成")
                except Exception as e:
                    print(f"⚠️ 处理器加载失败: {e}，但模型引擎已加载")
                    # 尝试从备用路径加载
                    try:
                        from transformers import AutoProcessor
                        self.processor = AutoProcessor.from_pretrained(self.vllm_model_path)
                        print("✅ 使用备用方式加载处理器成功")
                    except Exception as e2:
                        print(f"❌ 备用处理器加载也失败: {e2}")
                        print("🔄 处理器加载失败，但vLLM引擎可用")
                        # 如果vLLM引擎成功但处理器失败，仍然可以使用基本功能
                
                return True
            else:
                print(f"⚠️ 未启用vLLM，尝试回退模式")
                
        except Exception as e:
            error_msg = self._handle_vllm_error(e, "模型加载")
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            # 最后的回退尝试
            print("🔄 最后尝试回退模式...")
            if self._try_fallback_mode():
                return True
            
            # 所有方法都失败了，提供完整的诊断报告
            self._provide_compatibility_report()
            return False

    def extract_video_features(self, video_path: str, extract_audio: bool = False, extract_frame: bool = False):
        """从视频中提取音频和最后一帧"""
        features = {}
        
        if extract_audio:
            try:
                audio, sr = librosa.load(video_path, sr=16000)
                features['audio'] = audio
                print(f"提取音频成功: {len(audio)} samples at {sr}Hz")
            except Exception as e:
                print(f"音频提取失败: {e}")
        
        if extract_frame:
            try:
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                ret, frame = cap.read()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    features['last_frame'] = image
                    print("提取视频最后一帧成功")
                else:
                    print("提取帧失败")
                cap.release()
            except Exception as e:
                print(f"帧提取失败: {e}")
        
        return features

    def save_temp_file(self, data: Union[np.ndarray, Image.Image, str], file_type: str, suffix: str = "") -> str:
        """保存临时文件并返回路径"""
        timestamp = int(time.time())
        filename = f"{file_type}_{timestamp}{suffix}"
        filepath = self.temp_dir / filename
        
        try:
            if file_type == "audio" and isinstance(data, np.ndarray):
                sf.write(str(filepath), data, 16000)
            elif file_type == "image" and isinstance(data, Image.Image):
                data.save(str(filepath))
            elif file_type == "video" and isinstance(data, str):
                # 如果是文件路径，复制到临时目录
                import shutil
                shutil.copy2(data, str(filepath))
            else:
                print(f"⚠️ 不支持的文件类型: {type(data)}")
                return ""
                
            print(f"✅ 临时文件保存成功: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"❌ 临时文件保存失败: {e}")
            return ""

    def _diagnose_system_state(self):
        """诊断系统状态"""
        diagnostic_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'model_loaded': hasattr(self, 'vllm_model'),
            'processor_loaded': hasattr(self, 'processor'),
        }
        
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    diagnostic_info[f'gpu_{i}_name'] = props.name
                    diagnostic_info[f'gpu_{i}_memory'] = f"{props.total_memory / 1024**3:.1f}GB"
                    diagnostic_info[f'gpu_{i}_allocated'] = f"{torch.cuda.memory_allocated(i) / 1024**3:.1f}GB"
            except Exception as e:
                diagnostic_info['gpu_error'] = str(e)
        
        return diagnostic_info

    def _handle_vllm_error(self, error: Exception, context: str) -> str:
        """处理vLLM错误并提供诊断信息"""
        error_str = str(error).lower()
        diagnostic = self._diagnose_system_state()
        
        # 常见错误的处理建议
        if 'out of memory' in error_str or 'cuda out of memory' in error_str:
            suggestion = (
                "💾 显存不足错误:\n"
                f"  - 当前显存使用: {diagnostic.get('gpu_0_allocated', 'unknown')}\n"
                "  - 建议降低gpu_memory_utilization参数\n"
                "  - 或减少max_model_len和max_num_seqs参数"
            )
        elif 'import' in error_str or 'module' in error_str:
            suggestion = (
                "📦 模块导入错误:\n"
                "  - 检查vLLM是否正确安装: pip install vllm\n"
                "  - 确认vLLM版本支持Qwen2.5-Omni\n"
                "  - 检查Python环境和依赖"
            )
        elif 'omnillmengine' in error_str:
            suggestion = (
                "🤖 OmniLLMEngine错误:\n"
                "  - 检查模型路径是否正确\n"
                "  - 确认vLLM版本支持OmniLLMEngine\n"
                "  - 尝试使用enforce_eager=True"
            )
        elif 'tokenizer' in error_str or 'processor' in error_str:
            suggestion = (
                "🔤 处理器错误:\n"
                "  - 检查模型路径下是否有完整的tokenizer文件\n"
                "  - 确认trust_remote_code=True\n"
                "  - 尝试重新下载模型文件"
            )
        else:
            suggestion = (
                "⚠️ 通用错误:\n"
                "  - 检查模型路径和权限\n"
                "  - 确认所有依赖已正确安装\n"
                "  - 尝试使用更保守的配置参数"
            )
        
        diagnostic_str = "\n".join([f"  {k}: {v}" for k, v in diagnostic.items()])
        
        return f"""
❌ {context}失败:
错误信息: {error}

{suggestion}

🔍 系统诊断:
{diagnostic_str}
"""

    def process_multimodal(self, 
                          text_input: Optional[str],
                          image_input: Optional[Image.Image],
                          audio_input: Optional[str],
                          video_input: Optional[str],
                          system_prompt: str,
                          max_tokens: int,
                          extract_video_audio: bool,
                          extract_video_frame: bool,
                          using_mm_info_audio: bool,
                          enable_audio_output: bool = False):
        """处理多模态输入"""
        
        if not hasattr(self, 'vllm_model'):
            diagnostic = self._diagnose_system_state()
            error_msg = f"模型未加载 - 诊断信息: {diagnostic}"
            return MultimodalResponse(
                status="❌ 模型未加载",
                response_text="",
                processing_time=0,
                peak_memory=0,
                error=error_msg
            )
        
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        extracted_audio = None
        extracted_frame = None
        generated_audio = None
        
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            ]
            
            user_content = []
            
            # 添加文本
            if text_input and text_input.strip():
                user_content.append({"type": "text", "text": text_input.strip()})
            
            # 处理视频
            if video_input:
                if extract_video_audio or extract_video_frame:
                    print(f"🎬 开始提取视频特征...")
                    features = self.extract_video_features(
                        video_input, 
                        extract_audio=extract_video_audio, 
                        extract_frame=extract_video_frame
                    )
                    
                    if 'audio' in features:
                        # 保存提取的音频
                        temp_audio_path = self.save_temp_file(features['audio'], "audio", ".wav")
                        if temp_audio_path:
                            user_content.append({"type": "audio", "audio": features['audio']})
                            extracted_audio = temp_audio_path
                            print(f"✅ 音频已提取并保存: {temp_audio_path}")
                    
                    if 'last_frame' in features:
                        # 保存提取的图像
                        temp_image_path = self.save_temp_file(features['last_frame'], "image", ".png")
                        if temp_image_path:
                            user_content.append({"type": "image", "image": features['last_frame']})
                            extracted_frame = temp_image_path
                            print(f"✅ 图像已提取并保存: {temp_image_path}")
                else:
                    user_content.append({"type": "video", "video": video_input, "using_mm_info_audio": using_mm_info_audio})
            
            # 处理图像
            if image_input:
                user_content.append({"type": "image", "image": image_input})
            
            # 处理音频
            if audio_input:
                try:
                    audio_data, _ = librosa.load(audio_input, sr=16000)
                    user_content.append({"type": "audio", "audio": audio_data})
                except Exception as e:
                    print(f"音频处理失败: {e}")
            
            # 如果没有任何内容，使用默认
            if not user_content:
                user_content.append({"type": "text", "text": "Hello"})
            
            messages.append({"role": "user", "content": user_content})
            
            print(f"📝 构建的消息包含 {len(user_content)} 个内容项")
            
            # 应用聊天模板
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"📄 生成的prompt长度: {len(text_prompt)} 字符")
            
            # 处理多模态信息 - 使用vLLM的fetch函数
            audios, images, videos = [], [], []
            
            # 处理音频
            if audio_input:
                try:
                    audio_data, _ = librosa.load(audio_input, sr=16000)
                    audios.append(audio_data)
                except Exception as e:
                    print(f"音频处理失败: {e}")
            
            # 处理图像
            if image_input:
                try:
                    # 使用vLLM的fetch_image
                    from vllm.multimodal.processing_omni import fetch_image
                    image_data = fetch_image({'image': image_input})
                    images.append(image_data)
                except Exception as e:
                    print(f"图像处理失败: {e}")
            
            # 处理视频
            if video_input:
                try:
                    # 使用vLLM的fetch_video
                    from vllm.multimodal.processing_omni import fetch_video
                    video_data = fetch_video({'video': video_input})
                    videos.append(video_data)
                except Exception as e:
                    print(f"视频处理失败: {e}")
            
            # 处理从视频提取的音频和图像
            if extracted_audio:
                try:
                    audio_data, _ = librosa.load(extracted_audio, sr=16000)
                    audios.append(audio_data)
                except Exception as e:
                    print(f"提取音频处理失败: {e}")
            
            if extracted_frame:
                try:
                    from vllm.multimodal.processing_omni import fetch_image
                    image_data = fetch_image({'image': extracted_frame})
                    images.append(image_data)
                except Exception as e:
                    print(f"提取图像处理失败: {e}")
            
            print(f"📊 多模态处理结果: audios={len(audios)}, images={len(images)}, videos={len(videos)}")
            
            # 使用OmniLLMEngine生成
            print("🚀 开始使用OmniLLMEngine生成回答...")
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=0.0,
                top_k=-1,
                top_p=1.0,
                repetition_penalty=1.1,
                max_tokens=max_tokens,
                detokenize=True,
                seed=0
            )
            
            # 构建TextPrompt输入
            from vllm.inputs import TextPrompt
            multi_modal_data = {}
            if audios:
                multi_modal_data["audio"] = audios
            if images:
                multi_modal_data["image"] = images
            if videos:
                multi_modal_data["video"] = videos
            multi_modal_data["use_audio_in_video"] = using_mm_info_audio
            
            prompt = TextPrompt(
                prompt=text_prompt,
                multi_modal_data=multi_modal_data,
            )
            
            # 生成回答
            request_id = str(uuid.uuid4())
            output_queue = self.vllm_model.add_request(
                request_id,
                prompt,
                sampling_params
            )
            
            # 获取输出
            response_text = ""
            try:
                while True:
                    output = output_queue.get(timeout=30)  # 30秒超时
                    if output is None:
                        break
                    
                    if hasattr(output, 'outputs') and len(output.outputs) > 0:
                        if output.outputs[0].text:
                            response_text = output.outputs[0].text
                            print(f"📤 OmniLLMEngine生成输出: {response_text[:100]}...")
                        else:
                            response_text = "OmniLLMEngine生成失败"
                            print("❌ OmniLLMEngine生成失败")
                        break
            except queue.Empty:
                print("⚠️ 输出获取超时")
                response_text = "OmniLLMEngine生成超时"
            
            # 处理音频输出（vLLM模式下暂不支持音频生成）
            if enable_audio_output:
                print("⚠️ vLLM模式下音频输出暂不支持")
                generated_audio = None
            
            # 计算统计信息
            processing_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            
            print(f"✅ 处理完成，时间: {processing_time:.2f}秒，峰值显存: {peak_memory:.1f}MB")
            
            return MultimodalResponse(
                status=f"✅ 处理完成 - vLLM模式",
                response_text=response_text,
                extracted_audio=extracted_audio,
                extracted_frame=extracted_frame,
                generated_audio=generated_audio,
                processing_time=processing_time,
                peak_memory=peak_memory
            )
            
        except Exception as e:
            error_msg = self._handle_vllm_error(e, "多模态处理")
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            return MultimodalResponse(
                status="❌ 处理失败",
                response_text="",
                processing_time=time.time() - start_time,
                peak_memory=torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
                error=error_msg
            )

    def process_multimodal_streaming(self, 
                                   text_input: Optional[str],
                                   image_input: Optional[Image.Image],
                                   audio_input: Optional[str],
                                   video_input: Optional[str],
                                   system_prompt: str,
                                   max_tokens: int,
                                   extract_video_audio: bool,
                                   extract_video_frame: bool,
                                   using_mm_info_audio: bool,
                                   enable_audio_output: bool = False):
        """流式处理多模态输入"""
        
        if not hasattr(self, 'vllm_model'):
            yield json.dumps({
                "status": "❌ 模型未加载",
                "response_text": "",
                "error": "模型未加载，请先启动服务"
            })
            return
        
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        extracted_audio = None
        extracted_frame = None
        
        try:
            # 前期处理
            yield json.dumps({"status": "🔄 开始处理...", "response_text": ""})
            
            # 构建消息
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            ]
            
            user_content = []
            
            # 添加文本
            if text_input and text_input.strip():
                user_content.append({"type": "text", "text": text_input.strip()})
            
            # 处理视频
            if video_input:
                if extract_video_audio or extract_video_frame:
                    yield json.dumps({"status": "🎬 提取视频特征...", "response_text": ""})
                    
                    features = self.extract_video_features(
                        video_input, 
                        extract_audio=extract_video_audio, 
                        extract_frame=extract_video_frame
                    )
                    
                    if 'audio' in features:
                        temp_audio_path = self.save_temp_file(features['audio'], "audio", ".wav")
                        if temp_audio_path:
                            user_content.append({"type": "audio", "audio": features['audio']})
                            extracted_audio = temp_audio_path
                    
                    if 'last_frame' in features:
                        temp_image_path = self.save_temp_file(features['last_frame'], "image", ".png")
                        if temp_image_path:
                            user_content.append({"type": "image", "image": features['last_frame']})
                            extracted_frame = temp_image_path
                        
                    yield json.dumps({"status": "✅ 视频特征提取完成", "response_text": ""})
                else:
                    user_content.append({"type": "video", "video": video_input})
            
            # 处理图像
            if image_input:
                user_content.append({"type": "image", "image": image_input})
            
            # 处理音频
            if audio_input:
                try:
                    audio_data, _ = librosa.load(audio_input, sr=16000)
                    user_content.append({"type": "audio", "audio": audio_data})
                except Exception as e:
                    print(f"音频处理失败: {e}")
            
            if not user_content:
                user_content.append({"type": "text", "text": "Hello"})
            
            messages.append({"role": "user", "content": user_content})
            
            yield json.dumps({"status": "📝 构建多模态输入...", "response_text": ""})
            
            # 应用聊天模板
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 处理多模态信息 - 使用vLLM的fetch函数
            audios, images, videos = [], [], []
            
            # 处理音频
            if audio_input:
                try:
                    audio_data, _ = librosa.load(audio_input, sr=16000)
                    audios.append(audio_data)
                except Exception as e:
                    print(f"音频处理失败: {e}")
            
            # 处理图像
            if image_input:
                try:
                    # 使用vLLM的fetch_image
                    from vllm.multimodal.processing_omni import fetch_image
                    image_data = fetch_image({'image': image_input})
                    images.append(image_data)
                except Exception as e:
                    print(f"图像处理失败: {e}")
            
            # 处理视频
            if video_input:
                try:
                    # 使用vLLM的fetch_video
                    from vllm.multimodal.processing_omni import fetch_video
                    video_data = fetch_video({'video': video_input})
                    videos.append(video_data)
                except Exception as e:
                    print(f"视频处理失败: {e}")
            
            # 处理从视频提取的音频和图像
            if extracted_audio:
                try:
                    audio_data, _ = librosa.load(extracted_audio, sr=16000)
                    audios.append(audio_data)
                except Exception as e:
                    print(f"提取音频处理失败: {e}")
            
            if extracted_frame:
                try:
                    from vllm.multimodal.processing_omni import fetch_image
                    image_data = fetch_image({'image': extracted_frame})
                    images.append(image_data)
                except Exception as e:
                    print(f"提取图像处理失败: {e}")
            
            yield json.dumps({"status": "🚀 开始流式生成...", "response_text": ""})
            
            # 使用OmniLLMEngine流式生成
            sampling_params = SamplingParams(
                temperature=0.0,
                top_k=-1,
                top_p=1.0,
                repetition_penalty=1.1,
                max_tokens=max_tokens,
                detokenize=True,
                seed=0
            )
            
            # 构建TextPrompt输入
            from vllm.inputs import TextPrompt
            multi_modal_data = {}
            if audios:
                multi_modal_data["audio"] = audios
            if images:
                multi_modal_data["image"] = images
            if videos:
                multi_modal_data["video"] = videos
            multi_modal_data["use_audio_in_video"] = using_mm_info_audio
            
            prompt = TextPrompt(
                prompt=text_prompt,
                multi_modal_data=multi_modal_data,
            )
            
            # 流式生成
            response_text = ""
            request_id = str(uuid.uuid4())
            output_queue = self.vllm_model.add_request(
                request_id,
                prompt,
                sampling_params
            )
            
            # 流式获取输出
            while True:
                try:
                    output = output_queue.get(timeout=30)  # 30秒超时
                    if output is None:
                        break
                    
                    if hasattr(output, 'outputs') and len(output.outputs) > 0:
                        if output.outputs[0].text:
                            new_text = output.outputs[0].text
                            if new_text.strip():
                                response_text += new_text
                                processing_time = time.time() - start_time
                                status = f"📡 流式生成中... ({processing_time:.1f}s)"
                                yield json.dumps({
                                    "status": status,
                                    "response_text": response_text,
                                    "extracted_audio": extracted_audio,
                                    "extracted_frame": extracted_frame
                                })
                        
                        if output.finished:
                            break
                except queue.Empty:
                    print("⚠️ 流式输出超时")
                    break
            
            # 最终结果
            processing_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            
            final_status = f"""✅ 流式生成完成!
⏱️ 总时间: {processing_time:.2f}秒
💾 峰值显存: {peak_memory:.1f}MB
📝 输出长度: {len(response_text)} 字符"""
            
            yield json.dumps({
                "status": final_status,
                "response_text": response_text,
                "extracted_audio": extracted_audio,
                "extracted_frame": extracted_frame,
                "processing_time": processing_time,
                "peak_memory": peak_memory
            })
            
        except Exception as e:
            error_msg = self._handle_vllm_error(e, "流式多模态处理")
            print(error_msg)
            yield json.dumps({
                "status": "❌ 处理失败",
                "response_text": "",
                "error": error_msg
            })

# 创建FastAPI应用
app = FastAPI(
    title="Qwen2.5-Omni vLLM API服务",
    description="基于vLLM的Qwen2.5-Omni多模态模型API服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建模型服务实例
model_service = ModelService()

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    print("🚀 正在启动Qwen2.5-Omni vLLM API服务...")
    success = model_service.load_model()
    if success:
        print("✅ 模型加载成功，服务启动完成")
    else:
        print("❌ 模型加载失败，服务启动失败")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Qwen2.5-Omni vLLM API服务",
        "status": "running",
        "model_loaded": hasattr(model_service, 'vllm_model')
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": hasattr(model_service, 'vllm_model'),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.post("/process", response_model=MultimodalResponse)
async def process_multimodal(
    text_input: Optional[str] = Form(None),
    system_prompt: str = Form("You are a helpful AI assistant."),
    max_tokens: int = Form(512),
    extract_video_audio: bool = Form(False),
    extract_video_frame: bool = Form(False),
    using_mm_info_audio: bool = Form(False),
    enable_audio_output: bool = Form(False),
    image_input: Optional[UploadFile] = File(None),
    audio_input: Optional[UploadFile] = File(None),
    video_input: Optional[UploadFile] = File(None)
):
    """处理多模态输入"""
    
    # 处理文件上传
    image_data = None
    audio_data = None
    video_data = None
    
    if image_input:
        try:
            image_data = Image.open(image_input.file)
            print(f"✅ 图像上传成功: {image_input.filename}")
        except Exception as e:
            print(f"❌ 图像处理失败: {e}")
    
    if audio_input:
        try:
            # 保存音频文件
            temp_audio_path = model_service.temp_dir / f"upload_audio_{int(time.time())}.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_input.file.read())
            audio_data = str(temp_audio_path)
            print(f"✅ 音频上传成功: {audio_input.filename}")
        except Exception as e:
            print(f"❌ 音频处理失败: {e}")
    
    if video_input:
        try:
            # 保存视频文件
            temp_video_path = model_service.temp_dir / f"upload_video_{int(time.time())}.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(video_input.file.read())
            video_data = str(temp_video_path)
            print(f"✅ 视频上传成功: {video_input.filename}")
        except Exception as e:
            print(f"❌ 视频处理失败: {e}")
    
    # 调用模型服务
    result = model_service.process_multimodal(
        text_input=text_input,
        image_input=image_data,
        audio_input=audio_data,
        video_input=video_data,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        extract_video_audio=extract_video_audio,
        extract_video_frame=extract_video_frame,
        using_mm_info_audio=using_mm_info_audio,
        enable_audio_output=enable_audio_output
    )
    
    return result

@app.post("/process_streaming")
async def process_multimodal_streaming(
    text_input: Optional[str] = Form(None),
    system_prompt: str = Form("You are a helpful AI assistant."),
    max_tokens: int = Form(512),
    extract_video_audio: bool = Form(False),
    extract_video_frame: bool = Form(False),
    using_mm_info_audio: bool = Form(False),
    enable_audio_output: bool = Form(False),
    image_input: Optional[UploadFile] = File(None),
    audio_input: Optional[UploadFile] = File(None),
    video_input: Optional[UploadFile] = File(None)
):
    """流式处理多模态输入"""
    
    # 处理文件上传（与process函数相同）
    image_data = None
    audio_data = None
    video_data = None
    
    if image_input:
        try:
            image_data = Image.open(image_input.file)
        except Exception as e:
            print(f"❌ 图像处理失败: {e}")
    
    if audio_input:
        try:
            temp_audio_path = model_service.temp_dir / f"upload_audio_{int(time.time())}.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_input.file.read())
            audio_data = str(temp_audio_path)
        except Exception as e:
            print(f"❌ 音频处理失败: {e}")
    
    if video_input:
        try:
            temp_video_path = model_service.temp_dir / f"upload_video_{int(time.time())}.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(video_input.file.read())
            video_data = str(temp_video_path)
        except Exception as e:
            print(f"❌ 视频处理失败: {e}")
    
    # 返回流式响应
    def generate():
        for result in model_service.process_multimodal_streaming(
            text_input=text_input,
            image_input=image_data,
            audio_input=audio_data,
            video_input=video_data,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            extract_video_audio=extract_video_audio,
            extract_video_frame=extract_video_frame,
            using_mm_info_audio=using_mm_info_audio,
            enable_audio_output=enable_audio_output
        ):
            yield f"data: {result}\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/files/{file_type}/{filename}")
async def get_file(file_type: str, filename: str):
    """获取临时文件"""
    file_path = model_service.temp_dir / file_type / filename
    if file_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(file_path))
    else:
        raise HTTPException(status_code=404, detail="文件未找到")

if __name__ == "__main__":
    # 启动服务
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"🚀 启动vLLM API服务: http://{host}:{port}")
    print(f"📚 API文档: http://{host}:{port}/docs")
    
    uvicorn.run(
        "vllm_model_service:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

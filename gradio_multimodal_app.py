#!/usr/bin/env python3
"""
Qwen2.5-Omni 多模态Gradio界面
支持视频、语音、图像、文本等不同模态的组合输入
支持GPTQ量化版本以降低显存占用
"""

import os
import time
import logging
from typing import List, Optional, Tuple, Dict, Any

import torch
import numpy as np
import librosa
import cv2
from PIL import Image
import gradio as gr
from dotenv import load_dotenv
import soundfile as sf

# 导入qwen-omni-utils  
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor
from transformers import Qwen2_5OmniForConditionalGeneration

# GPTQ相关导入
from transformers.utils.hub import cached_file
from gptqmodel import GPTQModel
from gptqmodel.models.base import BaseGPTQModel
from gptqmodel.models.auto import MODEL_MAP, SUPPORTED_MODELS
from gptqmodel.models._const import CPU
from huggingface_hub import snapshot_download

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义GPTQ模型类
class Qwen25OmniThinkerGPTQ(BaseGPTQModel):
    loader = Qwen2_5OmniForConditionalGeneration
    base_modules = [
        "thinker.model.embed_tokens", 
        "thinker.model.norm", 
        "token2wav", 
        "thinker.audio_tower", 
        "thinker.model.rotary_emb",
        "thinker.visual", 
        "talker"
    ]
    pre_lm_head_norm_module = "thinker.model.norm"
    require_monkeypatch = False
    layers_node = "thinker.model.layers"
    layer_type = "Qwen2_5OmniDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
   
    def pre_quantize_generate_hook_start(self):
        # 定义move_to函数
        def move_to(tensor, device):
            if hasattr(tensor, 'to'):
                return tensor.to(device)
            return tensor
        
        self.thinker.visual = move_to(self.thinker.visual, device=self.quantize_config.device)
        self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        # 定义move_to函数
        def move_to(tensor, device):
            if hasattr(tensor, 'to'):
                return tensor.to(device)
            return tensor
        
        self.thinker.visual = move_to(self.thinker.visual, device=CPU)
        self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

# 注册GPTQ模型
MODEL_MAP["qwen2_5_omni"] = Qwen25OmniThinkerGPTQ
SUPPORTED_MODELS.extend(["qwen2_5_omni"])

# 定义speaker字典加载补丁
@classmethod
def patched_from_config(cls, config, *args, **kwargs):
    kwargs.pop("trust_remote_code", None)
    
    model = cls._from_config(config, **kwargs)
    
    # 尝试加载speaker字典
    try:
        spk_path = cached_file(
            config._name_or_path if hasattr(config, '_name_or_path') else "Qwen/Qwen2.5-Omni",
            "spk_dict.pt",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if spk_path is not None:
            model.load_speakers(spk_path)
            print("✅ Speaker字典加载成功")
        else:
            print("⚠️ Speaker字典未找到，跳过加载")
    except Exception as e:
        print(f"⚠️ Speaker字典加载失败: {e}")
    
    return model

# 应用补丁
Qwen2_5OmniForConditionalGeneration.from_config = patched_from_config


class MultimodalProcessor:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.getenv("MODEL_PATH", "/home/caden/models/Qwen2.5-Omni-3B")
        self.temp_files = []
        self.use_gptq = os.getenv("USE_GPTQ", "false").lower() == "true"
        self.gptq_model_path = os.getenv("GPTQ_MODEL_PATH", "")
        
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
        
    def _analyze_module_memory_usage(self, stage: str):
        """分析Qwen2.5-Omni模型各个模块的显存/内存占用"""
        if self.model is None:
            return
            
        print(f"\n🔍 {stage} - 模块级资源占用分析:")
        
        # 定义主要模块及其路径
        main_modules = {
            "thinker": "thinker",
            "talker": "talker", 
            "token2wav": "token2wav",
            "thinker.model": "thinker.model",
            "thinker.visual": "thinker.visual",
            "thinker.audio_tower": "thinker.audio_tower",
            "thinker.lm_head": "thinker.lm_head"
        }
        
        total_params = 0
        total_trainable_params = 0
        
        for module_name, module_path in main_modules.items():
            try:
                # 获取模块对象
                module = self.model
                for attr in module_path.split('.'):
                    if hasattr(module, attr):
                        module = getattr(module, attr)
                    else:
                        module = None
                        break
                
                if module is not None:
                    # 统计参数数量
                    module_params = sum(p.numel() for p in module.parameters())
                    module_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    
                    # 统计显存占用
                    module_gpu_memory = 0
                    module_cpu_memory = 0
                    
                    for param in module.parameters():
                        if param.device.type == 'cuda':
                            module_gpu_memory += param.numel() * param.element_size()
                        else:
                            module_cpu_memory += param.numel() * param.element_size()
                    
                    # 统计缓冲区显存
                    for buffer in module.buffers():
                        if buffer.device.type == 'cuda':
                            module_gpu_memory += buffer.numel() * buffer.element_size()
                        else:
                            module_cpu_memory += buffer.numel() * buffer.element_size()
                    
                    total_params += module_params
                    total_trainable_params += module_trainable_params
                    
                    print(f"   📊 {module_name}:")
                    print(f"      参数数量: {module_params:,} ({module_params/1e6:.2f}M)")
                    print(f"      可训练参数: {module_trainable_params:,} ({module_trainable_params/1e6:.2f}M)")
                    print(f"      GPU显存: {module_gpu_memory/1024**3:.3f} GB")
                    print(f"      CPU内存: {module_cpu_memory/1024**3:.3f} GB")
                    
                    # 显示设备位置
                    if hasattr(module, 'device'):
                        print(f"      设备位置: {module.device}")
                    else:
                        # 尝试从参数推断设备
                        device = next(module.parameters()).device if list(module.parameters()) else "unknown"
                        print(f"      设备位置: {device}")
                        
                else:
                    print(f"   ❌ {module_name}: 模块不存在")
                    
            except Exception as e:
                print(f"   ❌ {module_name}: 分析失败 - {e}")
        
        print(f"\n   📈 总计:")
        print(f"      总参数: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"      可训练参数: {total_trainable_params:,} ({total_trainable_params/1e6:.2f}M)")
        
        # 显示模型总显存占用
        if torch.cuda.is_available():
            model_gpu_memory = sum(p.numel() * p.element_size() for p in self.model.parameters() if p.device.type == 'cuda')
            model_gpu_memory += sum(b.numel() * b.element_size() for b in self.model.buffers() if b.device.type == 'cuda')
            print(f"      模型总GPU显存: {model_gpu_memory/1024**3:.3f} GB")
        
        print()
        
    def _monitor_inference_memory(self, modality_name: str):
        """监控推理过程中特定模态的显存占用"""
        class MemoryMonitor:
            def __init__(self, name, processor):
                self.name = name
                self.processor = processor
                self.start_memory = 0
                self.peak_memory = 0
                self.activation_memory = 0
                self.hooks = []
                
            def __enter__(self):
                if torch.cuda.is_available():
                    # 记录开始时的显存状态
                    torch.cuda.empty_cache()
                    self.start_memory = torch.cuda.memory_allocated()
                    self.peak_memory = self.start_memory
                    
                    # 设置钩子来监控激活值
                    self._setup_hooks()
                    
                    print(f"🔍 开始监控 {self.name} 模态推理显存...")
                    print(f"   初始显存: {self.start_memory / 1024**3:.3f} GB")
                
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if torch.cuda.is_available():
                    # 移除钩子
                    self._remove_hooks()
                    
                    # 计算最终统计
                    final_memory = torch.cuda.memory_allocated()
                    total_peak = torch.cuda.max_memory_allocated()
                    
                    # 计算激活显存（推理过程中的临时显存）
                    self.activation_memory = max(0, total_peak - self.start_memory)
                    
                    print(f"🔍 {self.name} 模态推理完成 - 显存统计:")
                    print(f"   初始显存: {self.start_memory / 1024**3:.3f} GB")
                    print(f"   最终显存: {final_memory / 1024**3:.3f} GB")
                    print(f"   峰值显存: {total_peak / 1024**3:.3f} GB")
                    print(f"   激活显存: {self.activation_memory / 1024**3:.3f} GB")
                    print(f"   推理增量: {(final_memory - self.start_memory) / 1024**3:.3f} GB")
                    print()
                    
                    # 重置峰值统计
                    torch.cuda.reset_peak_memory_stats()
                
            def _setup_hooks(self):
                """设置钩子来监控激活值"""
                def forward_hook(module, input, output):
                    if torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated()
                        self.peak_memory = max(self.peak_memory, current_memory)
                        
                        # 计算激活显存
                        if hasattr(output, 'numel'):
                            if hasattr(output, 'element_size'):
                                activation_size = output.numel() * output.element_size()
                            else:
                                activation_size = output.numel() * 4  # 假设float32
                            
                            if output.device.type == 'cuda':
                                self.activation_memory = max(self.activation_memory, activation_size)
                
                # 为关键模块添加钩子
                if hasattr(self.processor.model, 'thinker'):
                    if hasattr(self.processor.model.thinker, 'visual'):
                        self.hooks.append(self.processor.model.thinker.visual.register_forward_hook(forward_hook))
                    if hasattr(self.processor.model.thinker, 'audio_tower'):
                        self.hooks.append(self.processor.model.thinker.audio_tower.register_forward_hook(forward_hook))
                    if hasattr(self.processor.model.thinker, 'model'):
                        self.hooks.append(self.processor.model.thinker.model.register_forward_hook(forward_hook))
                
                if hasattr(self.processor.model, 'talker'):
                    self.hooks.append(self.processor.model.talker.register_forward_hook(forward_hook))
                
                if hasattr(self.processor.model, 'token2wav'):
                    self.hooks.append(self.processor.model.token2wav.register_forward_hook(forward_hook))
            
            def _remove_hooks(self):
                """移除所有钩子"""
                for hook in self.hooks:
                    hook.remove()
                self.hooks.clear()
        
        return MemoryMonitor(modality_name, self)
        
    def load_model(self):
        """加载模型和处理器"""
        try:
            if self.use_gptq:
                print(f"🚀 正在加载GPTQ量化模型: {self.gptq_model_path}")
                
                # 定义设备映射
                device_map = {
                    "thinker.model": "cuda", 
                    "thinker.lm_head": "cuda", 
                    "thinker.visual": "cpu",  
                    "thinker.audio_tower": "cpu",  
                    "talker": "cuda",  
                    "token2wav": "cuda",  
                }
                
                # 加载GPTQ模型
                self.model = GPTQModel.load(
                    self.gptq_model_path, 
                    device_map=device_map, 
                    torch_dtype=torch.float16,   
                    attn_implementation="flash_attention_2"
                )
                print("📦 GPTQ模型加载完成")
                
                # 统计显存占用
                if torch.cuda.is_available():
                    self._print_memory_usage("GPTQ模型加载完成后")
                
                # 分析各模块资源占用
                self._analyze_module_memory_usage("GPTQ模型加载完成后")
                
                # 加载处理器
                self.processor = Qwen2_5OmniProcessor.from_pretrained(self.gptq_model_path)
                print("处理器加载完成")
                
                return "✅ GPTQ量化模型加载成功"
            else:
                print(f"正在加载标准模型: {self.model_path}")
                # 标准模式但尝试优化显存
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                print("📦 模型加载完成")

                # 统计显存占用
                if torch.cuda.is_available():
                    self._print_memory_usage("标准模型加载完成后")
                
                # 分析各模块资源占用
                self._analyze_module_memory_usage("标准模型加载完成后")
                
                self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
                print("处理器加载完成")
                return "✅ 标准模型加载成功"
            
        except Exception as e:
            error_msg = f"❌ 模型加载失败: {e}"
            print(error_msg)
            return error_msg

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

    def process_multimodal(self, 
                          text_input: str,
                          image_input: Optional[Image.Image],
                          audio_input: Optional[str],
                          video_input: Optional[str],
                          system_prompt: str,
                          max_tokens: int,
                          extract_video_audio: bool,
                          extract_video_frame: bool,
                          using_mm_info_audio: bool,
                          enable_streaming: bool = False,
                          enable_audio_output: bool = False):
        """处理多模态输入"""
        
        if self.model is None:
            return "❌ 请先加载模型", "", None, None, 0, 0, None
        
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
                        user_content.append({"type": "audio", "audio": features['audio']})
                        # 保存提取的音频供显示
                        temp_audio_path = f"temp_extracted_audio_{int(time.time())}.wav"
                        sf.write(temp_audio_path, features['audio'], 16000)
                        extracted_audio = temp_audio_path
                        print(f"✅ 音频已提取并保存: {temp_audio_path}")
                    
                    if 'last_frame' in features:
                        user_content.append({"type": "image", "image": features['last_frame']})
                        # 保存提取的图像供显示
                        extracted_frame = features['last_frame']
                        print(f"✅ 图像已提取")
                else:
                    # add args to qwen-mm-info-utils
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
            
            # 处理多模态信息
            audios, images, videos = process_mm_info(messages, use_audio_in_video=using_mm_info_audio)
            if audios and not audio_input:
                print("开启使用mm_info_audio提取的音频且没有音频输入，将使用mm_info_audio的音频")
                messages.append({"role": "user", "content": [{"type": "audio", "audio": audios}]})
            print(f"📊 多模态处理结果: audios={len(audios) if audios else 0}, images={len(images) if images else 0}, videos={len(videos) if videos else 0}")
            
            # 处理输入
            inputs = self.processor(
                text=text_prompt, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                use_audio_in_video= not audios.is_empty(),
                padding=True
            )
            
            print(f"🔧 输入tensor形状: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs.items()]}")
            
            # 根据模型类型处理输入
            if self.use_gptq:
                # GPTQ模型：直接发送到CUDA并转换为float16
                inputs = inputs.to('cuda').to(torch.float16)
                print("🔧 GPTQ模型输入已发送到CUDA并转换为float16")
            else:
                # 标准模型：获取模型设备
                device = next(self.model.parameters()).device
                inputs = inputs.to(device).to(self.model.dtype)
                print(f"🔧 标准模型输入已发送到设备: {device}")
            
            print("🚀 开始生成回答...")
            
            # 根据输入模态类型确定监控名称
            modality_types = []
            if text_input and text_input.strip():
                modality_types.append("文本")
            if image_input:
                modality_types.append("图像")
            if audio_input:
                modality_types.append("音频")
            if video_input:
                modality_types.append("视频")
            
            modality_name = "+".join(modality_types) if modality_types else "纯文本"
            
            if enable_streaming:
                # 流式生成
                print("📡 使用流式输出...")
                response_text = ""
                
                # 获取输入长度用于后续过滤
                input_length = inputs['input_ids'].shape[1]
                
                with torch.no_grad():
                    # 使用流式生成
                    from transformers import TextIteratorStreamer
                    import threading
                    
                    streamer = TextIteratorStreamer(
                        self.processor.tokenizer, 
                        skip_prompt=True,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    
                    generation_kwargs = dict(
                        inputs=inputs,
                        streamer=streamer,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        use_audio_in_video=True,
                        return_audio=enable_audio_output,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
                    # 使用显存监控器监控推理过程
                    with self._monitor_inference_memory(f"{modality_name}-流式推理"):
                        # 在单独线程中生成
                        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
                        thread.start()
                        
                        # 流式读取生成的文本
                        for new_text in streamer:
                            response_text += new_text
                            # 实时更新可以在这里处理，但gradio需要特殊处理
                        
                        thread.join()
                
                print(f"📡 流式生成完成，总长度: {len(response_text)} 字符")
                
                # 流式模式下音频输出处理
                if enable_audio_output and self.use_gptq:
                    print("⚠️ GPTQ模型在流式模式下音频输出暂不支持，请使用标准模式")
                    generated_audio = None
                
            else:
                # 标准生成
                with torch.no_grad():
                    # 使用显存监控器监控推理过程
                    with self._monitor_inference_memory(f"{modality_name}-标准推理"):
                        output = self.model.generate(
                            **inputs, 
                            max_new_tokens=max_tokens,
                            use_audio_in_video=True,
                            return_audio=enable_audio_output,  # 根据参数决定是否返回音频
                            pad_token_id=self.processor.tokenizer.eos_token_id
                        )
                
                print(f"📤 生成输出形状: {output[0].shape if enable_audio_output else output.shape}")
                
                # 处理音频输出 - 根据模型类型处理
                if enable_audio_output:
                    try:
                        if self.use_gptq:
                            # GPTQ模型：检查output[2]是否为音频
                            if len(output) > 1 and output[1] is not None:
                                audio_data = output[1]
                                if hasattr(audio_data, 'reshape'):
                                    audio_data = audio_data.reshape(-1).detach().cpu().numpy()
                                else:
                                    audio_data = audio_data.cpu().numpy()
                                
                                audio_filename = f"generated_audio_gptq_{int(time.time())}.wav"
                                sf.write(audio_filename, audio_data, 24000)
                                generated_audio = audio_filename
                                print(f"🎵 GPTQ音频已生成并保存: {audio_filename}")
                            else:
                                print("⚠️ GPTQ模型未返回音频")
                                generated_audio = None
                        else:
                            # 标准模型：检查output.audio
                            if hasattr(output, 'audio') and output.audio is not None:
                                audio_filename = f"generated_audio_{int(time.time())}.wav"
                                sf.write(audio_filename, output.audio.cpu().numpy(), 24000)
                                generated_audio = audio_filename
                                print(f"🎵 标准模型音频已生成并保存: {audio_filename}")
                            else:
                                print("⚠️ 标准模型未返回音频")
                                generated_audio = None
                    except Exception as e:
                        print(f"音频保存失败: {e}")
                        generated_audio = None
                
                # 解码响应 - 根据模型类型处理
                if self.use_gptq:
                    # GPTQ模型：output[0]是文本输出
                    if len(output) > 1:
                        response_text = self.processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    elif len(output) == 1:
                        response_text = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    else:
                        response_text = "GPTQ模型输出格式异常"
                else:
                    # 标准模型：直接解码
                    response_text = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
                print(f"📝 原始回答长度: {len(response_text)} 字符")
                
                # 提取实际的回答部分 (去掉输入的prompt)
                # 找到assistant回答的开始位置
                if "<|im_start|>assistant" in response_text:
                    response_text = response_text.split("<|im_start|>assistant")[-1].strip()
                elif "assistant\n" in response_text:
                    response_text = response_text.split("assistant\n")[-1].strip()
                
                # 清理结束符号
                if response_text.endswith("<|im_end|>"):
                    response_text = response_text[:-10].strip()
            
            print(f"✅ 清理后回答: {response_text[:100]}...")
            
            # 计算统计信息
            processing_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            
            # 构建输出信息
            info_text = f"""
📊 **处理统计**
- ⏱️ 处理时间: {processing_time:.2f}秒
- 💾 峰值显存: {peak_memory:.1f}MB
- 🔤 最大Token数: {max_tokens}
- 📝 输出Token数: {len(output[0]) - len(inputs['input_ids'][0])}
- 🎯 系统提示: {system_prompt[:50]}...

📋 **输入内容**
- 文本输入: {'✅' if text_input else '❌'}
- 图像输入: {'✅' if image_input else '❌'}  
- 音频输入: {'✅' if audio_input else '❌'}
- 视频输入: {'✅' if video_input else '❌'}
- 提取音轨: {'✅' if extract_video_audio and extracted_audio else '❌'}
- 提取帧: {'✅' if extract_video_frame and extracted_frame else '❌'}
            """
            
            # 构建详细的处理信息
            status_info = f"""✅ 处理完成 - {'流式' if enable_streaming else '标准'}模式
⏱️ 处理时间: {processing_time:.2f}秒
💾 峰值显存: {peak_memory:.1f}MB"""

            return status_info, response_text, extracted_audio, extracted_frame, processing_time, peak_memory, generated_audio
            
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, "", None, None, 0, 0, None

    def process_multimodal_streaming(self, 
                                   text_input: str,
                                   image_input: Optional[Image.Image],
                                   audio_input: Optional[str],
                                   video_input: Optional[str],
                                   system_prompt: str,
                                   max_tokens: int,
                                   extract_video_audio: bool,
                                   extract_video_frame: bool,
                                   using_mm_info_audio: bool,
                                   enable_audio_output: bool = False):
        """流式处理多模态输入 - 使用生成器返回逐步更新"""
        
        if self.model is None:
            yield "❌ 请先加载模型", "", None, None, 0, 0, None
            return
        
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        extracted_audio = None
        extracted_frame = None
        generated_audio = None
        
        try:
            # 前期处理 - 和普通处理相同
            yield "🔄 开始处理...", "", None, None, 0, 0, None
            
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
                    yield "🎬 提取视频特征...", "", None, None, 0, 0, None
                    
                    features = self.extract_video_features(
                        video_input, 
                        extract_audio=extract_video_audio, 
                        extract_frame=extract_video_frame
                    )
                    
                    if 'audio' in features:
                        user_content.append({"type": "audio", "audio": features['audio']})
                        temp_audio_path = f"temp_extracted_audio_{int(time.time())}.wav"
                        sf.write(temp_audio_path, features['audio'], 16000)
                        extracted_audio = temp_audio_path
                    
                    if 'last_frame' in features:
                        user_content.append({"type": "image", "image": features['last_frame']})
                        extracted_frame = features['last_frame']
                        
                    yield "✅ 视频特征提取完成", "", extracted_audio, extracted_frame, 0, 0, None
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
            
            yield "📝 构建多模态输入...", "", extracted_audio, extracted_frame, 0, 0, None
            
            # 应用聊天模板
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 处理多模态信息
            audios, images, videos = process_mm_info(messages, use_audio_in_video=using_mm_info_audio)
            
            # 处理输入
            inputs = self.processor(
                text=text_prompt, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True
            )
            
            device = next(self.model.parameters()).device
            inputs = inputs.to(device).to(self.model.dtype)
            
            yield "🚀 开始流式生成...", "", extracted_audio, extracted_frame, 0, 0, None
            
            # 流式生成
            from transformers import TextIteratorStreamer
            import threading
            
            streamer = TextIteratorStreamer(
                self.processor.tokenizer, 
                skip_prompt=True,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_audio_in_video=True,
                return_audio=enable_audio_output,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            
            # 在单独线程中生成
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 流式输出
            response_text = ""
            for new_text in streamer:
                if new_text.strip():  # 忽略空白token
                    response_text += new_text
                    processing_time = time.time() - start_time
                    status = f"📡 流式生成中... ({processing_time:.1f}s)"
                    yield status, response_text, extracted_audio, extracted_frame, processing_time, 0, None
            
            thread.join()
            
            # 处理音频输出（流式模式下音频在最后生成）
            if enable_audio_output:
                try:
                    # 这里需要重新生成一次来获取音频，或者修改流式逻辑
                    # 为了简化，我们暂时在流式模式下不返回音频
                    generated_audio = None
                    print("📡 流式模式下音频输出暂不支持")
                except Exception as e:
                    print(f"流式音频处理失败: {e}")
                    generated_audio = None
            
            # 最终结果
            processing_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            
            final_status = f"""✅ 流式生成完成!
⏱️ 总时间: {processing_time:.2f}秒
💾 峰值显存: {peak_memory:.1f}MB
📝 输出长度: {len(response_text)} 字符"""
            
            yield final_status, response_text, extracted_audio, extracted_frame, processing_time, peak_memory, generated_audio
            
        except Exception as e:
            error_msg = f"❌ 流式处理失败: {str(e)}"
            yield error_msg, "", extracted_audio, extracted_frame, 0, 0, None


# 创建处理器实例
processor = MultimodalProcessor()

# 构建Gradio界面
def create_interface():
    with gr.Blocks(title="Qwen2.5-Omni 多模态助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Qwen2.5-Omni 多模态智能助手
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ 模型控制")
                
                # 添加模型类型选择
                model_type = gr.Radio(
                    choices=["标准模型", "GPTQ量化模型"],
                    value="标准模型",
                    label="🚀 模型类型",
                    info="选择要使用的模型类型"
                )
                
                # 添加GPTQ模型路径输入
                gptq_model_path = gr.Textbox(
                    label="🔗 GPTQ模型路径",
                    value="/home/caden/workplace/models/Qwen2.5-Omni-7B-GPTQ-Int4",
                    placeholder="输入GPTQ模型路径或HuggingFace模型ID",
                    visible=False,
                    info="GPTQ量化模型的路径或HuggingFace模型ID"
                )
                
                load_btn = gr.Button("🔄 加载模型", variant="primary")
                model_status = gr.Textbox(
                    label="模型状态", 
                    value="⏳ 未加载", 
                    interactive=False
                )
                
                gr.Markdown("### ⚙️ 生成参数")
                system_prompt = gr.Textbox(
                    label="系统提示",
                    value="You are a helpful AI assistant.",
                    lines=2
                )
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=2048,
                    value=512,
                    step=50,
                    label="最大Token数"
                )
                
                gr.Markdown("### 🎬 视频处理选项")
                extract_video_audio = gr.Checkbox(
                    label="📢 提取视频音轨",
                    value=False,
                    info="将视频音轨提取为音频输入"
                )
                extract_video_frame = gr.Checkbox(
                    label="🖼️ 提取视频最后一帧",
                    value=False,
                    info="将视频最后一帧提取为图像输入"
                )
                using_mm_info_audio = gr.Checkbox(
                    label="🎵 使用mm_info提取音频",
                    value=False,
                    info="使用mm_info提取音频"
                )
                
                gr.Markdown("### ⚡ 输出模式")
                enable_streaming = gr.Checkbox(
                    label="📡 启用流式输出",
                    value=False,
                    info="实时逐步显示生成内容，提升交互体验"
                )
                enable_audio_output = gr.Checkbox(
                    label="🎵 启用语音输出",
                    value=False,
                    info="生成语音回答（如果模型支持）"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📝 多模态输入")
                
                text_input = gr.Textbox(
                    label="💬 文本输入",
                    placeholder="输入您的问题或指令...",
                    lines=3
                )
                
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="🖼️ 图像输入",
                            type="pil"
                        )
                        
                        audio_input = gr.Audio(
                            label="🎵 音频输入",
                            type="filepath"
                        )
                    
                    with gr.Column():
                        video_input = gr.Video(
                            label="🎬 视频输入"
                        )
                
                with gr.Row():
                    process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                    stream_btn = gr.Button("📡 流式处理", variant="secondary", size="lg", visible=False)
                clear_btn = gr.Button("🗑️ 清空", variant="secondary")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 💬 生成结果")
                output_text = gr.Textbox(
                    label="AI回答",
                    lines=8,
                    placeholder="生成的回答将显示在这里...",
                    interactive=False
                )
                
                # 显示提取的内容
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎵 提取的音频")
                        extracted_audio_display = gr.Audio(
                            label="从视频提取的音频",
                            visible=True,
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 🖼️ 提取的图像")
                        extracted_image_display = gr.Image(
                            label="从视频提取的最后一帧",
                            type="pil",
                            visible=True,
                            interactive=False
                        )
                
                # 显示生成的音频输出
                gr.Markdown("### 🎤 生成的语音回答")
                generated_audio_display = gr.Audio(
                    label="AI生成的语音回答",
                    visible=True,
                    interactive=False
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 处理信息")
                processing_info = gr.Textbox(
                    label="处理状态",
                    lines=8,
                    interactive=False,
                    value="等待处理..."
                )
        
        
        # 事件绑定
        
        # 模型类型选择控制GPTQ模型路径显示
        def update_gptq_path_visibility(model_type):
            if model_type == "GPTQ量化模型":
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)
        
        model_type.change(
            fn=update_gptq_path_visibility,
            inputs=[model_type],
            outputs=[gptq_model_path]
        )
        
        # 修改加载模型函数以支持模型类型选择
        def load_model_with_type(model_type, gptq_path):
            if model_type == "GPTQ量化模型":
                # 设置GPTQ模型路径
                processor.gptq_model_path = gptq_path
                processor.use_gptq = True
                print(f"🚀 准备加载GPTQ模型: {gptq_path}")
            else:
                # 使用标准模型
                processor.use_gptq = False
                print("🚀 准备加载标准模型")
            
            return processor.load_model()
        
        load_btn.click(
            fn=load_model_with_type,
            inputs=[model_type, gptq_model_path],
            outputs=model_status
        )
        
        def handle_process_standard(text_input, image_input, audio_input, video_input, system_prompt, max_tokens, 
                                   extract_video_audio, extract_video_frame, using_mm_info_audio, enable_streaming, enable_audio_output):
            """标准处理函数"""
            if enable_streaming:
                # 如果启用流式，给出提示
                return "📡 流式模式：请点击下面的流式处理按钮", "", None, None, None
            else:
                # 使用标准处理
                result = processor.process_multimodal(
                    text_input, image_input, audio_input, video_input,
                    system_prompt, max_tokens, extract_video_audio, extract_video_frame, using_mm_info_audio, False, enable_audio_output
                )
                return result[0], result[1], result[2], result[3], result[6]  # status, text, audio, image, generated_audio
        
        def handle_process_streaming(text_input, image_input, audio_input, video_input, system_prompt, max_tokens, 
                                   extract_video_audio, extract_video_frame, using_mm_info_audio, enable_audio_output):
            """流式处理函数"""
            for status, text, audio, image, time, memory, generated_audio in processor.process_multimodal_streaming(
                text_input, image_input, audio_input, video_input,
                system_prompt, max_tokens, extract_video_audio, extract_video_frame, using_mm_info_audio, enable_audio_output
            ):
                yield status, text, audio, image, generated_audio
        
        # 标准处理按钮
        process_btn.click(
            fn=handle_process_standard,
            inputs=[
                text_input, image_input, audio_input, video_input,
                system_prompt, max_tokens, extract_video_audio, extract_video_frame, using_mm_info_audio, enable_streaming, enable_audio_output
            ],
            outputs=[processing_info, output_text, extracted_audio_display, extracted_image_display, generated_audio_display]
        )
        
        # 流式按钮已在上面定义
        
        stream_btn.click(
            fn=handle_process_streaming,
            inputs=[
                text_input, image_input, audio_input, video_input,
                system_prompt, max_tokens, extract_video_audio, extract_video_frame, using_mm_info_audio, enable_audio_output
            ],
            outputs=[processing_info, output_text, extracted_audio_display, extracted_image_display, generated_audio_display]
        )
        
        # 根据流式开关控制按钮显示
        def update_buttons(enable_streaming):
            if enable_streaming:
                return gr.update(value="🚀 标准处理"), gr.update(visible=True)
            else:
                return gr.update(value="🚀 开始处理"), gr.update(visible=False)
        
        enable_streaming.change(
            fn=update_buttons,
            inputs=[enable_streaming],
            outputs=[process_btn, stream_btn]
        )
        
        def clear_all():
            return "", None, None, None, "", "等待处理...", None, None, False, None
        
        clear_btn.click(
            fn=clear_all,
            outputs=[text_input, image_input, audio_input, video_input, output_text, processing_info, extracted_audio_display, extracted_image_display, enable_streaming, generated_audio_display]
        )
    
    return demo


if __name__ == "__main__":
    # 创建界面
    demo = create_interface()
    
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True,
        show_error=True
    )
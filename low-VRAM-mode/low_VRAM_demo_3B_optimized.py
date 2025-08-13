from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration

from transformers import Qwen2_5OmniProcessor
from transformers.utils.hub import cached_file

# 尝试导入GPTQ相关模块，如果失败则使用备用方案
try:
    from gptqmodel import GPTQModel
    from gptqmodel.models.base import BaseGPTQModel
    from gptqmodel.models.auto import MODEL_MAP
    from gptqmodel.models._const import CPU, SUPPORTED_MODELS
    GPTQ_AVAILABLE = True
except ImportError:
    print("Warning: gptqmodel not available, using fallback approach")
    GPTQ_AVAILABLE = False
    # 创建简单的备用类
    class GPTQModel:
        @staticmethod
        def load(*args, **kwargs):
            return Qwen2_5OmniForConditionalGeneration.from_pretrained(*args, **kwargs)
    
    class BaseGPTQModel:
        pass
    
    MODEL_MAP = {}
    SUPPORTED_MODELS = []
    CPU = "cpu"

from huggingface_hub import snapshot_download

from qwen_omni_utils import process_mm_info
from typing import Any, Dict

import torch
import time
import soundfile as sf
import os
from dotenv import load_dotenv
import gc

# 定义move_to函数
def move_to(obj, device):
    """将对象移动到指定设备"""
    if hasattr(obj, 'to'):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to(item, device) for item in obj)
    else:
        return obj

# 加载.env文件
load_dotenv()

# 从.env文件获取模型路径，如果没有则使用默认值
model_path = os.getenv("MODEL_PATH", "/home/caden/workplace/models/Qwen2.5-Omni-3B")
if not model_path:
    print("Warning: MODEL_PATH not found in .env file, using default path")
    model_path = "/home/caden/workplace/models/Qwen2.5-Omni-3B"

# 检查模型路径是否存在
if not os.path.exists(model_path):
    print(f"Warning: Model path not found: {model_path}")
    print("Please set correct MODEL_PATH in .env file or ensure model exists")

if GPTQ_AVAILABLE:
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
            self.thinker.visual = move_to(self.thinker.visual, device=self.quantize_config.device)
            self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=self.quantize_config.device)

        def pre_quantize_generate_hook_end(self):
            self.thinker.visual = move_to(self.thinker.visual, device=CPU)
            self.thinker.audio_tower = move_to(self.thinker.audio_tower, device=CPU)

        def preprocess_dataset(self, sample: Dict) -> Dict:
            return sample
    

    MODEL_MAP["qwen2_5_omni"] = Qwen25OmniThinkerGPTQ
    SUPPORTED_MODELS.extend(["qwen2_5_omni"])

    @classmethod
    def patched_from_config(cls, config, *args, **kwargs):
        kwargs.pop("trust_remote_code", None)
        
        model = cls._from_config(config, **kwargs)
        
        # 检查本地模型路径中是否存在说话人字典
        spk_path = os.path.join(model_path, "spk_dict.pt")
        if os.path.exists(spk_path):
            model.load_speakers(spk_path)
        else:
            print(f"Warning: Speaker dictionary not found at {spk_path}")
            print("Model will run without speaker support")
        
        return model

    Qwen2_5OmniForConditionalGeneration.from_config = patched_from_config

# 使用更保守的device_map
device_map = {
    "thinker.model": "cuda", 
    "thinker.lm_head": "cuda", 
    "thinker.visual": "cpu",  
    "thinker.audio_tower": "cpu",  
    "talker": "cuda",  
    "token2wav": "cuda",  
}

print(f"Loading model from: {model_path}")

# 尝试加载模型
try:
    if GPTQ_AVAILABLE:
        model = GPTQModel.load(
            model_path, 
            device_map=device_map, 
            torch_dtype=torch.float16,   
            attn_implementation="eager"  # 使用eager而不是flash_attention_2
        )
        print("GPTQ Model loaded successfully!")
    else:
        # 如果 GPTQ 不可用，直接加载原模型
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.float16,
            attn_implementation="eager",  # 使用eager而不是flash_attention_2
            trust_remote_code=True
        )
        print("Original model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying to load without device_map...")
    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        print("Model loaded without device_map successfully!")
    except Exception as e2:
        print(f"Failed to load model: {e2}")
        raise

# 加载处理器
try:
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    print("Processor loaded successfully!")
except Exception as e:
    print(f"Error loading processor: {e}")
    raise

def video_inference(video_path, prompt, sys_prompt):
    """优化的视频推理函数，减少内存使用"""
    try:
        # 清理GPU内存
        torch.cuda.empty_cache()
        gc.collect()
        
        messages = [
            {"role": "system", "content": [
                    {"type": "text", "text": sys_prompt},
                ]},
            {"role": "user", "content": [
                    {"type": "video", "video": video_path},
                ]
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 处理多模态信息
        print("Processing multimodal information...")
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        
        # 清理中间变量
        del messages
        gc.collect()
        
        print("Processing inputs...")
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = inputs.to('cuda').to(model.dtype)
        
        # 清理中间变量
        del audios, images, videos, text
        gc.collect()
        torch.cuda.empty_cache()

        print("Generating response...")
        with torch.no_grad():  # 减少内存使用
            output = model.generate(
                **inputs, 
                use_audio_in_video=True, 
                return_audio=True,
                max_new_tokens=128,  # 减少生成长度
                do_sample=False,     # 使用贪婪解码
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # 清理输入
        del inputs
        gc.collect()
        torch.cuda.empty_cache()
        
        text = processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        audio = output[2]
        
        # 清理输出
        del output
        gc.collect()
        torch.cuda.empty_cache()
        
        return text, audio
        
    except Exception as e:
        print(f"Video inference error: {e}")
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
        raise


def text_inference(prompt, sys_prompt):
    """纯文本推理函数，用于测试模型基本功能"""
    try:
        messages = [
            {"role": "system", "content": [
                    {"type": "text", "text": sys_prompt},
                ]},
            {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                ]
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to('cuda').to(model.dtype)
        
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=256,  # 减少生成长度
                do_sample=False,     # 使用贪婪解码
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        response = processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # 清理内存
        del inputs, output
        gc.collect()
        torch.cuda.empty_cache()
        
        return response
        
    except Exception as e:
        print(f"Text inference error: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        raise

# 测试视频路径
video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

print("\n=== 开始测试 ===")

# 首先测试纯文本推理
print("\n1. 测试纯文本推理...")
text_prompt = "Hello! Please introduce yourself briefly."
try:
    text_response = text_inference(text_prompt, system_prompt)
    print(f"文本响应: {text_response[0]}")
except Exception as e:
    print(f"文本推理失败: {e}")

# 然后测试视频推理
print("\n2. 测试视频推理...")
try:
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    
    response, audio = video_inference(video_path, prompt=None, sys_prompt=system_prompt)
    
    end = time.time()
    peak_memory = torch.cuda.max_memory_allocated()

    # 保存音频
    audio_file_path = "./output_audio_3B_optimized.wav"
    sf.write(
        audio_file_path,
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )

    print(f"视频响应: {response[0]}")
    print(f"音频已保存到: {audio_file_path}")
    print(f"总推理时间: {end-start:.2f} 秒")
    print(f"峰值GPU显存使用: {peak_memory / 1024 / 1024:.2f} MB")

except Exception as e:
    print(f"视频推理失败: {e}")
    print("可能的原因:")
    print("1. 网络连接问题，无法下载测试视频")
    print("2. 模型不支持视频推理")
    print("3. 缺少必要的依赖库")
    print("4. 内存不足")

print("\n=== 测试完成 ===")




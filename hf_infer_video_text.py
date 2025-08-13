import torch
import gc
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 添加项目根目录到Python路径，以便导入qwen-omni-utils
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "qwen-omni-utils" / "src"))

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# 设置环境变量以优化内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 从环境变量加载模型路径，如果没有设置则使用默认路径
MODEL_PATH = os.getenv('MODEL_PATH', "/home/caden/workplace/models/Qwen2.5-Omni-3B")

def main():
    try:
        print("正在加载模型...")
        print(f"模型路径: {MODEL_PATH}")
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 启用内存优化设置
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float16,  # 使用半精度减少内存
            device_map="auto",
            low_cpu_mem_usage=True,  # 低CPU内存使用
        )
        
        print(f"模型加载后GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print("模型加载成功！")
        
        print("正在加载处理器...")
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
        print("处理器加载成功！")
        
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()

        # 视频输入配置
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
                    {"type": "text", "text": "请描述这个视频中发生了什么？"},
                ],
            },
        ]

        # 禁用音频处理以减少内存使用
        USE_AUDIO_IN_VIDEO = False

        try:
            print("正在处理视频输入...")
            print(f"处理前GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # Preparation for inference
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            
            print("正在编码输入...")
            inputs = processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            inputs = inputs.to(model.device).to(model.dtype)

            # 清理中间变量
            del audios, images, videos
            gc.collect()
            torch.cuda.empty_cache()

            print("正在生成文本输出...")
            print(f"生成前GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # 只生成文本，不生成音频
            with torch.no_grad():  # 禁用梯度计算以节省内存
                output = model.generate(
                    **inputs, 
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                    max_new_tokens=128,  # 限制生成长度以减少内存使用
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=processor.tokenizer.eos_token_id
                )

            print("正在解码输出...")
            print(f"输出类型: {type(output)}")
            
            # 处理输出格式
            if isinstance(output, tuple):
                text_ids = output[0]  # 提取文本ID部分
            else:
                text_ids = output
                
            # 将张量转换为CPU上的numpy数组
            if hasattr(text_ids, 'cpu'):
                text_ids_cpu = text_ids.cpu().numpy()
            else:
                text_ids_cpu = text_ids
                
            # 如果是2D数组，取第一行
            if len(text_ids_cpu.shape) > 1:
                text_ids_cpu = text_ids_cpu[0]
                
            print(f"处理后的text_ids: {text_ids_cpu}")
            
            # 解码文本
            text = processor.batch_decode([text_ids_cpu], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print("生成的文本:", text)
            
            print("完成！视频推理成功！")
            
        except Exception as e:
            print(f"推理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"模型加载过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理内存
        if 'model' in locals():
            del model
        if 'processor' in locals():
            del processor
        if 'inputs' in locals():
            del inputs
        if 'text_ids' in locals():
            del text_ids
        if 'output' in locals():
            del output
        gc.collect()
        torch.cuda.empty_cache()
        print("内存已清理")

if __name__ == "__main__":
    main()


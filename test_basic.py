import torch
import gc
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# 设置环境变量以优化内存分配
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

MODEL_PATH = "/home/caden/workplace/models/Qwen2.5-Omni-3B"

def test_basic_functionality():
    try:
        print("=== 基础功能测试 ===")
        print("正在加载模型...")
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 启用内存优化设置
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        print(f"模型加载后GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print("✅ 模型加载成功！")
        
        print("正在加载处理器...")
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
        print("✅ 处理器加载成功！")
        
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()

        # 测试基础文本生成
        print("\n正在测试基础文本生成...")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, how are you?"},
                ],
            },
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)

        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )

        # 处理输出
        if isinstance(output, tuple):
            text_ids = output[0]
        else:
            text_ids = output
            
        if hasattr(text_ids, 'cpu'):
            text_ids_cpu = text_ids.cpu().numpy()
        else:
            text_ids_cpu = text_ids
            
        if len(text_ids_cpu.shape) > 1:
            text_ids_cpu = text_ids_cpu[0]
            
        text_result = processor.batch_decode([text_ids_cpu], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print("生成的文本:", text_result)
        print("✅ 基础文本生成测试成功！")
        
        print("\n=== 测试完成 ===")
        print("当前可用的功能：")
        print("- ✅ 模型加载")
        print("- ✅ 处理器加载")
        print("- ✅ 基础文本生成")
        print("- ⚠️  视频处理（需要更多GPU内存）")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
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
        gc.collect()
        torch.cuda.empty_cache()
        print("\n内存已清理")

if __name__ == "__main__":
    test_basic_functionality()


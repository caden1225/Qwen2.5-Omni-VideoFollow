#!/usr/bin/env python3
"""
Nanonets-OCR-s 简单推理示例
使用transformers进行直接推理
"""

from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

def simple_ocr_inference(image_path, model_path="/home/caden/workplace/nanonets/Nanonets-OCR-s"):
    """简单的OCR推理函数"""
    
    print("正在加载模型...")
    
    # 加载模型和处理器
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, 
        torch_dtype="auto", 
        device_map="auto"
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    
    print("✓ 模型加载完成")
    
    # 加载图片
    image = Image.open(image_path)
    
    # 构建提示词
    prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    
    # 构建消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]},
    ]
    
    # 应用聊天模板
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 处理输入
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    print("正在进行OCR识别...")
    
    # 生成输出
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=4096, 
            do_sample=False
        )
    
    # 解码输出
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )
    
    return output_text[0]

def main():
    """主函数"""
    import os
    
    # 检查测试图片
    test_img_path = "test_document.png"
    if not os.path.exists(test_img_path):
        print(f"测试图片不存在: {test_img_path}")
        print("请先运行: python transformers_inference.py -c 创建示例图片")
        return
    
    print("=== Nanonets-OCR-s 简单推理示例 ===")
    print(f"使用测试图片: {test_img_path}")
    
    try:
        result = simple_ocr_inference(test_img_path)
        print("\n=== OCR识别结果 ===")
        print(result)
        print("\n=== 识别完成 ===")
    except Exception as e:
        print(f"推理失败: {e}")

if __name__ == "__main__":
    main() 
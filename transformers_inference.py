#!/usr/bin/env python3
"""
Nanonets-OCR-s Transformers 推理脚本
使用conda环境312，无需安装vLLM
"""

import os
import argparse
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

class NanonetsOCRInference:
    def __init__(self, model_path="/home/caden/workplace/nanonets/Nanonets-OCR-s"):
        """初始化OCR推理类"""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.processor = None
        
    def load_model(self):
        """加载模型"""
        try:
            print("正在加载模型...")
            
            # 检查模型路径
            if not os.path.exists(self.model_path):
                print(f"错误: 模型路径不存在: {self.model_path}")
                return False
            
            # 加载模型
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path, 
                torch_dtype="auto", 
                device_map="auto", 
                attn_implementation="flash_attention_2"
            )
            self.model.eval()
            
            # 加载tokenizer和processor
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            print("✓ 模型加载成功")
            return True
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            return False
    
    def ocr_page_with_nanonets_s(self, image_path, max_new_tokens=4096):
        """使用Nanonets-OCR-s进行OCR识别"""
        try:
            # 检查图片文件
            if not os.path.exists(image_path):
                print(f"图片文件不存在: {image_path}")
                return None
            
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
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 处理输入
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            # 生成输出
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=False
                )
            
            # 解码输出
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            return output_text[0]
            
        except Exception as e:
            print(f"OCR识别时出错: {e}")
            return None
    
    def test_with_sample_image(self, image_path):
        """使用示例图片进行测试"""
        print(f"正在测试图片: {image_path}")
        
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            return False
        
        # 进行OCR识别
        print("正在进行OCR识别...")
        result = self.ocr_page_with_nanonets_s(image_path)
        
        if result:
            print("\n=== OCR识别结果 ===")
            print(result)
            print("\n=== 识别完成 ===")
            return True
        else:
            print("OCR识别失败")
            return False

def create_sample_image():
    """创建一个示例图片用于测试"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # 创建一个简单的测试图片
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # 添加一些文本
        text = """测试文档

这是一个测试文档，用于验证Nanonets-OCR-s模型的功能。

表格示例:
| 项目 | 数量 | 价格 |
|------|------|------|
| 商品A | 10 | ¥100 |
| 商品B | 5 | ¥50 |

数学公式示例:
E = mc²

复选框示例:
☐ 选项1
☑ 选项2
☐ 选项3

页码: <page_number>1</page_number>
水印: <watermark>测试文档</watermark>
"""
        
        # 尝试使用默认字体
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # 绘制文本
        draw.text((50, 50), text, fill='black', font=font)
        
        # 保存图片
        sample_path = "test_document.png"
        img.save(sample_path)
        print(f"已创建示例图片: {sample_path}")
        return sample_path
        
    except Exception as e:
        print(f"创建示例图片时出错: {e}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Nanonets-OCR-s Transformers 推理脚本")
    parser.add_argument("--image", "-i", help="要测试的图片路径")
    parser.add_argument("--model-path", "-m", default="/home/caden/workplace/nanonets/Nanonets-OCR-s",
                       help="模型路径")
    parser.add_argument("--create-sample", "-c", action="store_true", 
                       help="创建示例图片进行测试")
    parser.add_argument("--max-tokens", "-t", type=int, default=4096,
                       help="最大生成token数")
    
    args = parser.parse_args()
    
    print("=== Nanonets-OCR-s Transformers 推理脚本 ===")
    
    # 创建推理实例
    ocr_inference = NanonetsOCRInference(model_path=args.model_path)
    
    # 加载模型
    if not ocr_inference.load_model():
        print("模型加载失败，请检查模型路径和依赖")
        return
    
    # 确定测试图片
    if args.create_sample:
        image_path = create_sample_image()
        if not image_path:
            print("创建示例图片失败")
            return
    elif args.image:
        image_path = args.image
    else:
        # 默认使用示例图片
        image_path = "test_document.png"
        if not os.path.exists(image_path):
            print("未找到测试图片，正在创建示例图片...")
            image_path = create_sample_image()
            if not image_path:
                print("创建示例图片失败")
                return
    
    # 进行测试
    success = ocr_inference.test_with_sample_image(image_path)
    
    if success:
        print("\n✓ 测试完成")
    else:
        print("\n✗ 测试失败")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Nanonets-OCR-s 使用示例
使用vLLM部署的模型进行OCR识别
"""

from openai import OpenAI
import base64
import os

def encode_image(image_path):
    """将图片编码为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ocr_page_with_nanonets_s(img_base64):
    """使用Nanonets-OCR-s进行OCR识别"""
    client = OpenAI(api_key="123", base_url="http://localhost:8000/v1")
    
    response = client.chat.completions.create(
        model="nanonets/Nanonets-OCR-s",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes.",
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=15000
    )
    return response.choices[0].message.content

def main():
    """主函数"""
    print("=== Nanonets-OCR-s 使用示例 ===")
    
    # 检查测试图片
    test_img_path = "test_document.png"
    if not os.path.exists(test_img_path):
        print(f"测试图片不存在: {test_img_path}")
        print("请先运行: python test_nanonets_ocr.py -c 创建示例图片")
        return
    
    print(f"使用测试图片: {test_img_path}")
    
    # 编码图片
    print("正在编码图片...")
    img_base64 = encode_image(test_img_path)
    
    # 进行OCR识别
    print("正在进行OCR识别...")
    try:
        result = ocr_page_with_nanonets_s(img_base64)
        print("\n=== OCR识别结果 ===")
        print(result)
        print("\n=== 识别完成 ===")
    except Exception as e:
        print(f"OCR识别失败: {e}")
        print("请确保vLLM服务器正在运行")

if __name__ == "__main__":
    main() 
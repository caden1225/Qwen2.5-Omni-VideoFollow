#!/usr/bin/env python3
"""
高级多模态API客户端演示
展示如何使用标准和流式API接口
"""

import os
import time
import asyncio
import json
import requests
import aiohttp
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查接口"""
    print("🔍 检查API健康状态...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API服务正常")
            print(f"   模型已加载: {data['model_loaded']}")
            print(f"   设备: {data['device']}")
            print(f"   模型路径: {data['model_path']}")
            return True
        else:
            print(f"❌ API服务异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到API服务: {e}")
        return False


def test_standard_text():
    """测试标准文本推理"""
    print("\n📝 测试标准文本推理...")
    
    try:
        data = {
            "text": "请用一段话介绍数学的重要性",
            "system_prompt": "You are a helpful AI assistant.",
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/multimodal/standard", json=data)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 标准推理成功 (耗时: {elapsed:.2f}s)")
            print(f"   处理时间: {result['processing_time']:.2f}s")
            print(f"   生成Token数: {result['tokens_generated']}")
            print(f"   峰值显存: {result['peak_memory_mb']:.1f}MB")
            print(f"\n📄 回答:")
            print("-" * 50)
            print(result['response'])
            print("-" * 50)
            return True
        else:
            print(f"❌ 标准推理失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 标准推理异常: {e}")
        return False


def test_standard_with_video():
    """测试带视频的标准推理"""
    print("\n🎬 测试带视频的标准推理...")
    
    video_path = "/home/caden/workspace/Qwen2.5-Omni-VideoFollow/math.mp4"
    
    if not os.path.exists(video_path):
        print("❌ 视频文件不存在，跳过测试")
        return False
    
    try:
        # 准备数据
        data = {
            "text": "请分析这个数学视频的内容",
            "system_prompt": "You are a helpful AI assistant.",
            "max_new_tokens": 200,
            "extract_video_audio": "true",
            "extract_video_frame": "true",
            "temperature": 0.3
        }
        
        # 准备文件
        files = {
            "videos": open(video_path, "rb")
        }
        
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/multimodal/standard", data=data, files=files)
        elapsed = time.time() - start_time
        
        files["videos"].close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 视频推理成功 (耗时: {elapsed:.2f}s)")
            print(f"   处理时间: {result['processing_time']:.2f}s")
            print(f"   生成Token数: {result['tokens_generated']}")
            print(f"   峰值显存: {result['peak_memory_mb']:.1f}MB")
            
            if result['extracted_audio_url']:
                print(f"🎵 提取音频: {API_BASE_URL}{result['extracted_audio_url']}")
            
            if result['extracted_image_url']:
                print(f"🖼️ 提取图像: {API_BASE_URL}{result['extracted_image_url']}")
            
            print(f"\n📄 回答:")
            print("-" * 50)
            print(result['response'])
            print("-" * 50)
            return True
        else:
            print(f"❌ 视频推理失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 视频推理异常: {e}")
        return False


async def test_streaming_text():
    """测试流式文本推理"""
    print("\n📡 测试流式文本推理...")
    
    try:
        data = {
            "text": "请详细介绍人工智能的发展历程",
            "system_prompt": "You are a helpful AI assistant.",
            "max_new_tokens": 300,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/multimodal/streaming",
                json=data,
                headers={"Accept": "text/event-stream"}
            ) as response:
                
                if response.status == 200:
                    print("✅ 流式连接建立成功")
                    print("📡 开始接收流式数据...\n")
                    
                    current_text = ""
                    start_time = time.time()
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            try:
                                chunk_data = json.loads(line[6:])  # 去掉 'data: '
                                event = chunk_data['event']
                                data_content = chunk_data['data']
                                
                                if event == 'start':
                                    print(f"🚀 {data_content['message']}")
                                
                                elif event == 'progress':
                                    print(f"⏳ {data_content['message']}")
                                
                                elif event == 'token':
                                    current_text = data_content['text']
                                    # 实时显示更新的文本 (只显示最后50字符避免刷屏)
                                    display_text = current_text[-50:] if len(current_text) > 50 else current_text
                                    print(f"\r📝 当前生成: ...{display_text}", end='', flush=True)
                                
                                elif event == 'done':
                                    elapsed = time.time() - start_time
                                    final_data = data_content
                                    print(f"\n✅ 流式生成完成 (耗时: {elapsed:.2f}s)")
                                    print(f"   处理时间: {final_data['processing_time']:.2f}s")
                                    print(f"   生成Token数: {final_data['tokens_generated']}")
                                    print(f"   峰值显存: {final_data.get('peak_memory_mb', 0):.1f}MB")
                                    print(f"\n📄 完整回答:")
                                    print("-" * 60)
                                    print(final_data['final_text'])
                                    print("-" * 60)
                                    return True
                                
                                elif event == 'error':
                                    print(f"\n❌ 流式处理出错: {data_content['error']}")
                                    return False
                                    
                            except json.JSONDecodeError as e:
                                print(f"⚠️ JSON解析错误: {e}")
                                continue
                else:
                    print(f"❌ 流式连接失败: {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ 流式推理异常: {e}")
        return False


async def test_streaming_with_video():
    """测试带视频的流式推理"""
    print("\n🎬📡 测试带视频的流式推理...")
    
    video_path = "/home/caden/workspace/Qwen2.5-Omni-VideoFollow/math.mp4"
    
    if not os.path.exists(video_path):
        print("❌ 视频文件不存在，跳过测试")
        return False
    
    try:
        # 准备表单数据
        data = aiohttp.FormData()
        data.add_field('text', '请详细分析这个数学视频，包括音频和视觉内容')
        data.add_field('system_prompt', 'You are a helpful AI assistant.')
        data.add_field('max_new_tokens', '250')
        data.add_field('extract_video_audio', 'true')
        data.add_field('extract_video_frame', 'true')
        data.add_field('temperature', '0.3')
        
        # 添加视频文件
        data.add_field('videos', open(video_path, 'rb'), filename='math.mp4')
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/multimodal/streaming",
                data=data,
                headers={"Accept": "text/event-stream"}
            ) as response:
                
                if response.status == 200:
                    print("✅ 视频流式连接建立成功")
                    print("📡 开始接收流式数据...\n")
                    
                    current_text = ""
                    start_time = time.time()
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            try:
                                chunk_data = json.loads(line[6:])
                                event = chunk_data['event']
                                data_content = chunk_data['data']
                                
                                if event == 'start':
                                    print(f"🚀 {data_content['message']}")
                                
                                elif event == 'progress':
                                    print(f"⏳ {data_content['message']}")
                                
                                elif event == 'audio_extracted':
                                    print(f"🎵 音频提取完成: {data_content['url']}")
                                    print(f"   时长: {data_content['duration']:.1f}秒")
                                
                                elif event == 'image_extracted':
                                    print(f"🖼️ 图像提取完成: {data_content['url']}")
                                    print(f"   尺寸: {data_content['size']}")
                                
                                elif event == 'token':
                                    current_text = data_content['text']
                                    display_text = current_text[-50:] if len(current_text) > 50 else current_text
                                    print(f"\r📝 当前生成: ...{display_text}", end='', flush=True)
                                
                                elif event == 'done':
                                    elapsed = time.time() - start_time
                                    final_data = data_content
                                    print(f"\n✅ 视频流式生成完成 (耗时: {elapsed:.2f}s)")
                                    print(f"   处理时间: {final_data['processing_time']:.2f}s")
                                    print(f"   生成Token数: {final_data['tokens_generated']}")
                                    
                                    if final_data.get('extracted_audio_url'):
                                        print(f"🎵 音频链接: {API_BASE_URL}{final_data['extracted_audio_url']}")
                                    
                                    if final_data.get('extracted_image_url'):
                                        print(f"🖼️ 图像链接: {API_BASE_URL}{final_data['extracted_image_url']}")
                                    
                                    print(f"\n📄 完整回答:")
                                    print("-" * 60)
                                    print(final_data['final_text'])
                                    print("-" * 60)
                                    return True
                                
                                elif event == 'error':
                                    print(f"\n❌ 视频流式处理出错: {data_content['error']}")
                                    return False
                                    
                            except json.JSONDecodeError as e:
                                continue
                else:
                    print(f"❌ 视频流式连接失败: {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ 视频流式推理异常: {e}")
        return False


async def main():
    """主函数 - 运行所有测试"""
    print("🚀 Qwen2.5-Omni 高级多模态API测试")
    print("=" * 60)
    
    # 检查API健康状态
    if not test_health():
        print("❌ API服务不可用，请先启动 advanced_multimodal_api.py")
        return
    
    results = {}
    
    # 测试1: 标准文本推理
    print("\n" + "="*60)
    results['standard_text'] = test_standard_text()
    
    # 测试2: 标准视频推理
    print("\n" + "="*60)
    results['standard_video'] = test_standard_with_video()
    
    # 测试3: 流式文本推理
    print("\n" + "="*60)
    results['streaming_text'] = await test_streaming_text()
    
    # 测试4: 流式视频推理
    print("\n" + "="*60)
    results['streaming_video'] = await test_streaming_with_video()
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结:")
    print(f"标准文本推理: {'✅' if results['standard_text'] else '❌'}")
    print(f"标准视频推理: {'✅' if results['standard_video'] else '❌'}")
    print(f"流式文本推理: {'✅' if results['streaming_text'] else '❌'}")  
    print(f"流式视频推理: {'✅' if results['streaming_video'] else '❌'}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    if success_count == total_count:
        print(f"\n🎉 所有测试通过 ({success_count}/{total_count})")
        print("💡 API功能完全正常，可以投入使用！")
    else:
        print(f"\n⚠️ 部分测试失败 ({success_count}/{total_count})")
        print("💡 请检查失败的测试项目")


if __name__ == "__main__":
    asyncio.run(main())
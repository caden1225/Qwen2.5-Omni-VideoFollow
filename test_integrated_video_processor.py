#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合版视频处理优化模块测试脚本
功能包括：
1. 大视频文件压缩优化测试
2. 通用功能验证测试
3. 自定义配置测试
4. 性能分析和报告生成
"""

import torch
import gc
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from video_processor_optimizer import (
    VideoProcessorOptimizer, 
    VideoOptimizationConfig, 
    VideoOptimizationPresets,
    load_config_from_file,
    save_config_to_file
)

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig

# 从环境变量加载模型路径
MODEL_PATH = os.getenv('MODEL_PATH', "/home/caden/workplace/models/Qwen2.5-Omni-3B")

def print_gpu_memory_usage(stage=""):
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"[{stage}] GPU内存 - 已分配: {allocated:.2f}GB, 已预留: {reserved:.2f}GB")

def load_model():
    """加载模型"""
    print("=== 加载模型 ===")
    
    try:
        config = Qwen2_5OmniConfig.from_pretrained(MODEL_PATH)
        config.enable_audio_output = False
        
        device_map = {
            "thinker.model": "cuda",
            "thinker.lm_head": "cuda",
            "thinker.visual": "cuda",
            "thinker.audio_tower": "cuda",
        }
        
        max_memory = {0: "8GB", "cpu": "16GB"}
        
        print("正在加载模型...")
        print_gpu_memory_usage("加载前")
        
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            config=config,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
        
        print_gpu_memory_usage("模型加载后")
        print("✅ 模型和处理器加载成功")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_video_with_optimization(video_path: str, preset_name: str, model, processor, detailed_output=True):
    """使用指定预设配置测试视频"""
    print(f"\n{'='*60}")
    print(f"🎬 测试视频: {os.path.basename(video_path)}")
    print(f"⚙️ 使用预设: {preset_name}")
    print(f"{'='*60}")
    
    try:
        # 获取预设配置
        config = VideoOptimizationPresets.get_preset(preset_name)
        print(f"📋 配置详情:")
        print(f"  - 帧数: {config.nframes}")
        print(f"  - 分辨率: {config.resized_width}x{config.resized_height}")
        print(f"  - 时间范围: {config.video_start}s - {config.video_end}s")
        print(f"  - 最大像素: {config.max_pixels:,}")
        print(f"  - 半精度: {config.use_half_precision}")
        
        # 创建优化器
        optimizer = VideoProcessorOptimizer(config)
        
        # 创建对话
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "你是一个视频分析助手， 请根据得到的信息完成任务。"}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "完成视频中的指令。"},
                ],
            },
        ]
        
        # 处理视频
        print(f"\n🔄 开始处理视频...")
        start_time = time.time()
        
        success, video_tensor, info = optimizer.process_video(video_path, conversation)
        
        if not success:
            print("❌ 视频处理失败")
            return False, {}, ""
        
        total_time = time.time() - start_time
        
        if detailed_output:
            print(f"\n✅ 视频处理成功！")
            print(f"📊 处理结果:")
            print(f"  - 原始文件大小: {info['file_size_mb']:.2f} MB")
            print(f"  - 最终张量形状: {info['final_shape']}")
            print(f"  - 最终内存占用: {info['final_memory_mb']:.2f} MB")
            print(f"  - 压缩比: {info['file_size_mb'] / info['final_memory_mb']:.0f}:1")
            print(f"  - 处理时间: {info['processing_time']:.2f}秒")
            print(f"  - 总耗时: {total_time:.2f}秒")
        else:
            print(f"✅ 视频处理成功！")
        
        print_gpu_memory_usage("视频处理后")
        
        # 应用聊天模板
        print(f"\n🔄 应用聊天模板...")
        text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # 处理输入
        print(f"🔄 处理模型输入...")
        inputs = processor(
            text=[text],
            images=None,  # 没有图片输入
            videos=[video_tensor],
            padding=True,
            return_tensors="pt",
        )
        
        # 确保张量类型正确
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if key in ["input_ids", "image_grid_thw", "video_grid_thw"]:
                    inputs[key] = value.to(model.device).long()
                elif "visual" in key:
                    inputs[key] = value.to(model.device).to(model.dtype)
                else:
                    inputs[key] = value.to(model.device).to(model.dtype)
        
        print_gpu_memory_usage("输入处理后")
        
        # 生成回复
        print(f"🔄 生成AI回复...")
        generation_start = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256 if detailed_output else 128,  # 根据输出模式调整token数量
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
        
        generation_time = time.time() - generation_start
        print_gpu_memory_usage("生成后")
        
        # 解码输出
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if detailed_output:
            print(f"\n🎉 推理成功！")
            print(f"🤖 AI回复:")
            print(f"{response}")
            print(f"\n⏱️ 生成耗时: {generation_time:.2f}秒")
        else:
            print(f"✅ 推理成功！")
            print(f"AI回复: {response}")
        
        # 清理
        optimizer.cleanup()
        
        return True, info, response
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, {}, ""

def test_custom_config(video_path: str, model, processor):
    """测试自定义配置"""
    print(f"\n{'='*60}")
    print(f"🔧 测试自定义配置")
    print(f"{'='*60}")
    
    try:
        # 创建自定义配置
        custom_config = VideoOptimizationConfig(
            nframes=6,                    # 6帧
            resized_height=168,           # 168x168分辨率
            resized_width=168,
            video_start=0.0,              # 从开始
            video_end=4.0,                # 前4秒
            max_pixels=256 * 28 * 28,     # 像素限制
            use_half_precision=True,
            enable_audio=False
        )
        
        print(f"📋 自定义配置:")
        print(f"  - 帧数: {custom_config.nframes}")
        print(f"  - 分辨率: {custom_config.resized_width}x{custom_config.resized_height}")
        print(f"  - 时间范围: {custom_config.video_start}s - {custom_config.video_end}s")
        print(f"  - 最大像素: {custom_config.max_pixels:,}")
        print(f"  - 半精度: {custom_config.use_half_precision}")
        
        # 创建优化器
        optimizer = VideoProcessorOptimizer(custom_config)
        
        # 创建对话
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "你是一个视频分析助手， 请根据得到的信息完成任务。"}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "完成视频中的指令。"},
                ],
            },
        ]
        
        # 处理视频
        print(f"\n🔄 开始处理视频...")
        success, video_tensor, info = optimizer.process_video(video_path, conversation)
        
        if not success:
            print("❌ 视频处理失败")
            return False
        
        print_gpu_memory_usage("视频处理后")
        
        # 应用聊天模板
        text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # 处理输入
        inputs = processor(
            text=[text],
            images=None,  # 没有图片输入
            videos=[video_tensor],
            padding=True,
            return_tensors="pt",
        )
        
        # 确保张量类型正确
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if key in ["input_ids", "image_grid_thw", "video_grid_thw"]:
                    inputs[key] = value.to(model.device).long()
                elif "visual" in key:
                    inputs[key] = value.to(model.device).to(model.dtype)
                else:
                    inputs[key] = value.to(model.device).to(model.dtype)
        
        print_gpu_memory_usage("输入处理后")
        
        # 生成回复
        print("🔄 生成AI回复...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
        
        print_gpu_memory_usage("生成后")
        
        # 解码输出
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"✅ 推理成功！")
        print(f"AI回复: {response}")
        print(f"处理信息: {info}")
        
        # 清理
        optimizer.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_presets_for_video(video_path: str, model, processor):
    """测试所有预设配置对指定视频的效果"""
    print(f"\n{'='*80}")
    print(f"🧪 全面测试视频: {os.path.basename(video_path)}")
    print(f"{'='*80}")
    
    # 获取所有预设配置
    presets = VideoOptimizationPresets.list_presets()
    
    results = {}
    
    for preset_name in presets:
        print(f"\n{'='*60}")
        print(f"🎯 测试预设: {preset_name}")
        print(f"{'='*60}")
        
        success, info, response = test_video_with_optimization(video_path, preset_name, model, processor)
        
        if success:
            results[preset_name] = {
                'success': True,
                'info': info,
                'response': response
            }
            print(f"✅ {preset_name} 预设测试成功")
        else:
            results[preset_name] = {
                'success': False,
                'info': {},
                'response': ""
            }
            print(f"❌ {preset_name} 预设测试失败")
        
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        print_gpu_memory_usage("清理后")
        
        # 等待一下再测试下一个
        time.sleep(2)
    
    return results

def test_basic_functionality(video_path: str, model, processor):
    """测试基本功能（使用部分预设配置）"""
    print(f"\n{'='*60}")
    print(f"🔍 基本功能测试: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    # 选择部分预设配置进行测试
    presets_to_test = ['extreme_low_memory', 'low_memory', 'balanced']
    
    results = {}
    
    for preset_name in presets_to_test:
        print(f"\n{'='*40}")
        print(f"🎯 测试预设: {preset_name}")
        print(f"{'='*40}")
        
        success, info, response = test_video_with_optimization(video_path, preset_name, model, processor, detailed_output=False)
        
        if success:
            results[preset_name] = {
                'success': True,
                'info': info,
                'response': response
            }
            print(f"✅ {preset_name} 预设测试成功")
        else:
            results[preset_name] = {
                'success': False,
                'info': {},
                'response': ""
            }
            print(f"❌ {preset_name} 预设测试失败")
        
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        print_gpu_memory_usage("清理后")
        
        # 等待一下再测试下一个
        time.sleep(1)
    
    return results

def print_summary_report(video_path: str, results: dict):
    """打印测试总结报告"""
    print(f"\n{'='*80}")
    print(f"📊 测试总结报告: {os.path.basename(video_path)}")
    print(f"{'='*80}")
    
    successful_presets = []
    failed_presets = []
    
    for preset_name, result in results.items():
        if result['success']:
            successful_presets.append(preset_name)
        else:
            failed_presets.append(preset_name)
    
    print(f"✅ 成功的预设配置 ({len(successful_presets)}/{len(results)}):")
    for preset in successful_presets:
        info = results[preset]['info']
        compression_ratio = info['file_size_mb'] / info['final_memory_mb']
        print(f"  - {preset}: 压缩比 {compression_ratio:.0f}:1, 内存 {info['final_memory_mb']:.2f}MB")
    
    if failed_presets:
        print(f"\n❌ 失败的预设配置 ({len(failed_presets)}):")
        for preset in failed_presets:
            print(f"  - {preset}")
    
    # 推荐最佳配置
    if successful_presets:
        best_preset = min(successful_presets, 
                         key=lambda x: results[x]['info']['final_memory_mb'])
        best_info = results[best_preset]['info']
        print(f"\n🏆 推荐配置: {best_preset}")
        print(f"  - 内存占用最低: {best_info['final_memory_mb']:.2f} MB")
        print(f"  - 压缩比: {best_info['file_size_mb'] / best_info['final_memory_mb']:.0f}:1")
        print(f"  - 处理时间: {best_info['processing_time']:.2f}秒")

def main():
    """主函数"""
    print("🚀 整合版视频处理优化测试脚本")
    print("="*80)
    print("功能包括：")
    print("1. 🎬 大视频文件压缩优化测试")
    print("2. 🔍 基本功能验证测试")
    print("3. 🔧 自定义配置测试")
    print("4. 📊 性能分析和报告生成")
    print("="*80)
    
    # 检查视频文件
    video_paths = [
        "./math.mp4",           # 82.76 MB - 之前无法运行的大文件
        "./math_last_3s.mp4",   # 3秒片段
        "./test_video.mp4"      # 32.82 MB - draw.mp4
    ]
    
    available_videos = []
    for path in video_paths:
        if os.path.exists(path):
            available_videos.append(path)
            file_size = os.path.getsize(path) / (1024 * 1024)
            print(f"📁 找到视频文件: {path} ({file_size:.2f} MB)")
        else:
            print(f"❌ 视频文件不存在: {path}")
    
    if not available_videos:
        print("❌ 未找到可用的视频文件，退出")
        return
    
    try:
        # 加载模型
        model, processor = load_model()
        
        if model is None or processor is None:
            print("❌ 模型加载失败，退出")
            return
        
        # 选择测试模式
        print(f"\n{'='*60}")
        print("🎯 请选择测试模式:")
        print("1. 🧪 全面测试模式 - 测试所有预设配置（适合大文件性能分析）")
        print("2. 🔍 基本测试模式 - 测试部分预设配置（适合功能验证）")
        print("3. 🔧 自定义配置测试 - 测试自定义优化参数")
        print("4. 🚀 全部测试 - 依次执行所有测试")
        print("="*60)
        
        # 这里可以根据需要修改测试模式
        test_mode = "4"  # 默认执行全部测试
        
        if test_mode == "1" or test_mode == "4":
            # 全面测试模式
            for video_path in available_videos:
                print(f"\n{'='*80}")
                print(f"🎬 开始全面测试视频: {os.path.basename(video_path)}")
                print(f"{'='*80}")
                
                # 测试所有预设配置
                results = test_all_presets_for_video(video_path, model, processor)
                
                # 打印总结报告
                print_summary_report(video_path, results)
                
                # 等待一下再测试下一个视频
                if len(available_videos) > 1:
                    print(f"\n⏳ 等待5秒后测试下一个视频...")
                    time.sleep(5)
        
        if test_mode == "2" or test_mode == "4":
            # 基本测试模式
            for video_path in available_videos:
                print(f"\n{'='*80}")
                print(f"🔍 开始基本功能测试: {os.path.basename(video_path)}")
                print(f"{'='*80}")
                
                # 测试部分预设配置
                results = test_basic_functionality(video_path, model, processor)
                
                # 打印总结报告
                print_summary_report(video_path, results)
                
                # 等待一下再测试下一个视频
                if len(available_videos) > 1:
                    print(f"\n⏳ 等待3秒后测试下一个视频...")
                    time.sleep(3)
        
        if test_mode == "3" or test_mode == "4":
            # 自定义配置测试
            print(f"\n{'='*80}")
            print(f"🔧 开始自定义配置测试")
            print(f"{'='*80}")
            
            # 使用第一个可用视频进行自定义配置测试
            test_video = available_videos[0]
            success = test_custom_config(test_video, model, processor)
            
            if success:
                print("✅ 自定义配置测试成功")
            else:
                print("❌ 自定义配置测试失败")
        
        print(f"\n🎉 所有测试完成！")
        print(f"📋 测试总结:")
        print(f"  - 测试视频数量: {len(available_videos)}")
        print(f"  - 可用预设配置: {len(VideoOptimizationPresets.list_presets())}")
        print(f"  - 成功处理大文件，内存占用大幅降低！")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
多模态API服务
支持视频、语音、图像、文本等不同模态的组合输入，输出文本
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import traceback

import torch
import numpy as np
import librosa
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

# 导入qwen-omni-utils
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor

# 尝试导入模型
try:
    from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration
    LOW_VRAM_MODE = True
except ImportError:
    from transformers import Qwen2_5OmniForConditionalGeneration
    LOW_VRAM_MODE = False

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen2.5-Omni 多模态API", version="1.0.0")

class MultimodalRequest(BaseModel):
    text: Optional[str] = None
    system_prompt: str = "You are a helpful AI assistant."
    max_new_tokens: int = 512
    extract_video_audio: bool = False  # 是否提取视频音轨
    extract_video_frame: bool = False  # 是否提取视频最后一帧


class MultimodalResponse(BaseModel):
    response: str
    processing_time: float
    peak_memory_mb: Optional[float] = None


class ModelManager:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.getenv("MODEL_PATH", "/home/caden/models/Qwen2.5-Omni-3B")
        self.load_model()

    def load_model(self):
        """加载模型和处理器"""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            if LOW_VRAM_MODE:
                # 使用低显存模式的设备映射
                device_map = {
                    "thinker.model": "cuda", 
                    "thinker.lm_head": "cuda", 
                    "thinker.visual": "cpu",  
                    "thinker.audio_tower": "cpu",  
                    "talker": "cuda",  
                    "token2wav": "cuda",  
                }
                
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    self.model_path,
                    device_map=device_map,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            else:
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(self.device)

            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
            logger.info("Model and processor loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def extract_video_features(self, video_path: str, extract_audio: bool = False, extract_frame: bool = False):
        """从视频中提取音频和帧"""
        features = {}
        
        if extract_audio:
            try:
                # 使用librosa提取音频
                audio, sr = librosa.load(video_path, sr=16000)
                features['audio'] = audio
                logger.info(f"Extracted audio from video: {len(audio)} samples at {sr}Hz")
            except Exception as e:
                logger.warning(f"Failed to extract audio: {e}")
        
        if extract_frame:
            try:
                # 使用opencv提取最后一帧
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                ret, frame = cap.read()
                
                if ret:
                    # 转换为PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    features['last_frame'] = image
                    logger.info("Extracted last frame from video")
                else:
                    logger.warning("Failed to extract frame")
                cap.release()
            except Exception as e:
                logger.warning(f"Failed to extract frame: {e}")
        
        return features

    def process_multimodal_input(self, 
                                text: str = None,
                                images: List[Image.Image] = None,
                                audios: List[np.ndarray] = None,
                                videos: List[str] = None,
                                system_prompt: str = "You are a helpful AI assistant.",
                                max_new_tokens: int = 512,
                                extract_video_audio: bool = False,
                                extract_video_frame: bool = False):
        """处理多模态输入并生成响应"""
        
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        try:
            # 构建消息格式
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            ]
            
            user_content = []
            
            # 添加文本
            if text:
                user_content.append({"type": "text", "text": text})
            
            # 处理视频
            if videos:
                for video_path in videos:
                    if extract_video_audio or extract_video_frame:
                        features = self.extract_video_features(
                            video_path, 
                            extract_audio=extract_video_audio, 
                            extract_frame=extract_video_frame
                        )
                        
                        # 如果提取了音频，添加到音频列表
                        if 'audio' in features:
                            if audios is None:
                                audios = []
                            audios.append(features['audio'])
                            # 添加音频内容
                            user_content.append({"type": "audio", "audio": features['audio']})
                        
                        # 如果提取了帧，添加到图像列表
                        if 'last_frame' in features:
                            if images is None:
                                images = []
                            images.append(features['last_frame'])
                            # 添加图像内容
                            user_content.append({"type": "image", "image": features['last_frame']})
                    else:
                        # 直接添加视频
                        user_content.append({"type": "video", "video": video_path})
            
            # 处理图像
            if images:
                for image in images:
                    user_content.append({"type": "image", "image": image})
            
            # 处理音频
            if audios:
                for audio in audios:
                    user_content.append({"type": "audio", "audio": audio})
            
            # 如果没有任何内容，添加默认文本
            if not user_content:
                user_content.append({"type": "text", "text": "Hello"})
                
            messages.append({"role": "user", "content": user_content})
            
            # 应用聊天模板
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 处理多模态信息
            audios_processed, images_processed, videos_processed = process_mm_info(messages, use_audio_in_video=True)
            
            # 处理输入
            inputs = self.processor(
                text=text_prompt, 
                audio=audios_processed, 
                images=images_processed, 
                videos=videos_processed, 
                return_tensors="pt", 
                padding=True
            )
            
            if LOW_VRAM_MODE:
                inputs = {k: v.to('cuda').to(self.model.dtype) if hasattr(v, 'to') else v for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device).to(self.model.dtype)
            
            # 生成响应
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, use_audio_in_video=True)
            response_text = self.processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            # 计算处理时间和内存使用
            processing_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None
            
            return MultimodalResponse(
                response=response_text,
                processing_time=processing_time,
                peak_memory_mb=peak_memory
            )
            
        except Exception as e:
            logger.error(f"Error processing multimodal input: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# 全局模型管理器
model_manager = None

@app.on_event("startup")
async def startup_event():
    global model_manager
    try:
        model_manager = ModelManager()
        logger.info("API服务启动成功")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


@app.post("/multimodal", response_model=MultimodalResponse)
async def multimodal_inference(
    request: MultimodalRequest,
    images: Optional[List[UploadFile]] = File(None),
    audios: Optional[List[UploadFile]] = File(None),  
    videos: Optional[List[UploadFile]] = File(None)
):
    """多模态推理接口"""
    try:
        # 处理上传的文件
        processed_images = []
        processed_audios = []
        processed_videos = []
        
        # 处理图像文件
        if images:
            for img_file in images:
                image = Image.open(img_file.file)
                processed_images.append(image)
        
        # 处理音频文件
        if audios:
            for audio_file in audios:
                # 保存临时文件
                temp_path = f"temp_audio_{int(time.time())}_{audio_file.filename}"
                with open(temp_path, "wb") as f:
                    f.write(await audio_file.read())
                
                # 加载音频
                audio_data, _ = librosa.load(temp_path, sr=16000)
                processed_audios.append(audio_data)
                
                # 清理临时文件
                os.remove(temp_path)
        
        # 处理视频文件
        if videos:
            for video_file in videos:
                # 保存临时文件
                temp_path = f"temp_video_{int(time.time())}_{video_file.filename}"
                with open(temp_path, "wb") as f:
                    f.write(await video_file.read())
                processed_videos.append(temp_path)
        
        # 调用模型处理
        response = model_manager.process_multimodal_input(
            text=request.text,
            images=processed_images if processed_images else None,
            audios=processed_audios if processed_audios else None,
            videos=processed_videos if processed_videos else None,
            system_prompt=request.system_prompt,
            max_new_tokens=request.max_new_tokens,
            extract_video_audio=request.extract_video_audio,
            extract_video_frame=request.extract_video_frame
        )
        
        # 清理临时视频文件
        for temp_path in processed_videos:
            try:
                os.remove(temp_path)
            except:
                pass
        
        return response
        
    except Exception as e:
        logger.error(f"API request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "model_loaded": model_manager is not None}


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Qwen2.5-Omni 多模态API服务",
        "version": "1.0.0",
        "endpoints": {
            "multimodal": "/multimodal - 多模态推理",
            "health": "/health - 健康检查"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
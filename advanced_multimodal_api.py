#!/usr/bin/env python3
"""
高级多模态API服务
提供与Gradio应用相同的完整功能，包括流式输出、视频音轨和图像提取
"""

import os
import sys
import time
import logging
import asyncio
import tempfile
import json
from typing import Dict, List, Optional, Union, AsyncGenerator
from pathlib import Path
import traceback

import torch
import numpy as np
import librosa
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import soundfile as sf

# 添加路径
sys.path.append('/home/caden/workspace/Qwen2.5-Omni-VideoFollow/qwen-omni-utils/src')

# 导入qwen-omni-utils
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen2.5-Omni 高级多模态API",
    version="2.0.0",
    description="支持标准和流式响应的完整多模态API服务"
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class StandardRequest(BaseModel):
    text: Optional[str] = None
    system_prompt: str = "You are a helpful AI assistant."
    max_new_tokens: int = Field(512, ge=1, le=2048)
    extract_video_audio: bool = False
    extract_video_frame: bool = False
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


class StreamingRequest(BaseModel):
    text: Optional[str] = None
    system_prompt: str = "You are a helpful AI assistant."
    max_new_tokens: int = Field(512, ge=1, le=2048)
    extract_video_audio: bool = False
    extract_video_frame: bool = False
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


# 响应模型
class StandardResponse(BaseModel):
    status: str
    response: str
    extracted_audio_url: Optional[str] = None
    extracted_image_url: Optional[str] = None
    processing_time: float
    peak_memory_mb: Optional[float] = None
    tokens_generated: int
    model_info: Dict[str, str]


class StreamingChunk(BaseModel):
    event: str  # "start", "progress", "audio_extracted", "image_extracted", "token", "done", "error"
    data: Dict[str, Union[str, int, float, bool]]


class ModelManager:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.getenv("MODEL_PATH", "/home/caden/models/Qwen2.5-Omni-3B")
        self.temp_files = []
        self.model_loaded = False

    async def load_model_async(self):
        """异步加载模型"""
        if self.model_loaded:
            return True
            
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # 在线程池中加载模型
            import concurrent.futures
            
            def load_model():
                model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
                return model, processor
            
            # 使用线程池执行器加载模型
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                self.model, self.processor = await loop.run_in_executor(executor, load_model)
            
            self.model_loaded = True
            logger.info("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False

    def extract_video_features(self, video_path: str, extract_audio: bool = False, extract_frame: bool = False):
        """从视频中提取音频和最后一帧"""
        features = {}
        temp_files = []
        
        if extract_audio:
            try:
                audio, sr = librosa.load(video_path, sr=16000)
                
                # 保存音频文件
                audio_filename = f"extracted_audio_{int(time.time() * 1000)}.wav"
                audio_path = f"/tmp/{audio_filename}"
                sf.write(audio_path, audio, 16000)
                
                features['audio'] = audio
                features['audio_path'] = audio_path
                features['audio_url'] = f"/audio/{audio_filename}"
                temp_files.append(audio_path)
                
                logger.info(f"✅ Audio extracted: {len(audio)} samples")
            except Exception as e:
                logger.warning(f"❌ Audio extraction failed: {e}")
        
        if extract_frame:
            try:
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                ret, frame = cap.read()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    
                    # 保存图像文件
                    image_filename = f"extracted_frame_{int(time.time() * 1000)}.jpg"
                    image_path = f"/tmp/{image_filename}"
                    image.save(image_path, quality=95)
                    
                    features['last_frame'] = image
                    features['image_path'] = image_path
                    features['image_url'] = f"/image/{image_filename}"
                    temp_files.append(image_path)
                    
                    logger.info(f"✅ Frame extracted: {image.size}")
                cap.release()
            except Exception as e:
                logger.warning(f"❌ Frame extraction failed: {e}")
        
        self.temp_files.extend(temp_files)
        return features

    async def process_multimodal_standard(self,
                                        text_input: str,
                                        images: List[Image.Image],
                                        audios: List[str],
                                        videos: List[str],
                                        system_prompt: str,
                                        max_new_tokens: int,
                                        extract_video_audio: bool,
                                        extract_video_frame: bool,
                                        temperature: float = 0.7,
                                        top_p: float = 0.9):
        """标准多模态处理"""
        
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        extracted_audio_url = None
        extracted_image_url = None
        
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
            if videos:
                for video_path in videos:
                    if extract_video_audio or extract_video_frame:
                        features = self.extract_video_features(
                            video_path, 
                            extract_audio=extract_video_audio, 
                            extract_frame=extract_video_frame
                        )
                        
                        if 'audio' in features:
                            user_content.append({"type": "audio", "audio": features['audio']})
                            extracted_audio_url = features['audio_url']
                        
                        if 'last_frame' in features:
                            user_content.append({"type": "image", "image": features['last_frame']})
                            extracted_image_url = features['image_url']
                    else:
                        user_content.append({"type": "video", "video": video_path})
            
            # 处理图像
            for image in images:
                user_content.append({"type": "image", "image": image})
            
            # 处理音频
            for audio_path in audios:
                try:
                    audio_data, _ = librosa.load(audio_path, sr=16000)
                    user_content.append({"type": "audio", "audio": audio_data})
                except Exception as e:
                    logger.warning(f"Audio processing failed: {e}")
            
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
            
            device = next(self.model.parameters()).device
            inputs = inputs.to(device).to(self.model.dtype)
            
            input_tokens = inputs['input_ids'].shape[1]
            
            # 生成响应
            with torch.no_grad():
                if temperature > 0.0:
                    output = self.model.generate(
                        **inputs, 
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        use_audio_in_video=True,
                        return_audio=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                else:
                    output = self.model.generate(
                        **inputs, 
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_audio_in_video=True,
                        return_audio=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
            
            response_text = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            # 清理输出文本
            if "<|im_start|>assistant" in response_text:
                response_text = response_text.split("<|im_start|>assistant")[-1].strip()
            elif "assistant\n" in response_text:
                response_text = response_text.split("assistant\n")[-1].strip()
            
            if response_text.endswith("<|im_end|>"):
                response_text = response_text[:-10].strip()
            
            # 计算统计信息
            processing_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None
            tokens_generated = output.shape[1] - input_tokens
            
            return StandardResponse(
                status="success",
                response=response_text,
                extracted_audio_url=extracted_audio_url,
                extracted_image_url=extracted_image_url,
                processing_time=processing_time,
                peak_memory_mb=peak_memory,
                tokens_generated=tokens_generated,
                model_info={
                    "model_path": self.model_path,
                    "device": str(device),
                    "dtype": str(self.model.dtype)
                }
            )
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return StandardResponse(
                status="error",
                response=f"Processing failed: {str(e)}",
                extracted_audio_url=extracted_audio_url,
                extracted_image_url=extracted_image_url,
                processing_time=time.time() - start_time,
                peak_memory_mb=None,
                tokens_generated=0,
                model_info={}
            )

    async def process_multimodal_streaming(self,
                                         text_input: str,
                                         images: List[Image.Image],
                                         audios: List[str],
                                         videos: List[str],
                                         system_prompt: str,
                                         max_new_tokens: int,
                                         extract_video_audio: bool,
                                         extract_video_frame: bool,
                                         temperature: float = 0.7,
                                         top_p: float = 0.9) -> AsyncGenerator[StreamingChunk, None]:
        """流式多模态处理"""
        
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        
        try:
            yield StreamingChunk(
                event="start",
                data={"message": "开始处理多模态输入", "timestamp": time.time()}
            )
            
            # 构建消息
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            ]
            
            user_content = []
            extracted_audio_url = None
            extracted_image_url = None
            
            # 添加文本
            if text_input and text_input.strip():
                user_content.append({"type": "text", "text": text_input.strip()})
            
            # 处理视频
            if videos:
                for video_path in videos:
                    if extract_video_audio or extract_video_frame:
                        yield StreamingChunk(
                            event="progress",
                            data={"message": "提取视频特征中...", "step": "video_extraction"}
                        )
                        
                        features = self.extract_video_features(
                            video_path, 
                            extract_audio=extract_video_audio, 
                            extract_frame=extract_video_frame
                        )
                        
                        if 'audio' in features:
                            user_content.append({"type": "audio", "audio": features['audio']})
                            extracted_audio_url = features['audio_url']
                            yield StreamingChunk(
                                event="audio_extracted",
                                data={
                                    "url": features['audio_url'],
                                    "duration": len(features['audio']) / 16000,
                                    "sample_rate": 16000
                                }
                            )
                        
                        if 'last_frame' in features:
                            user_content.append({"type": "image", "image": features['last_frame']})
                            extracted_image_url = features['image_url']
                            yield StreamingChunk(
                                event="image_extracted",
                                data={
                                    "url": features['image_url'],
                                    "size": features['last_frame'].size
                                }
                            )
                    else:
                        user_content.append({"type": "video", "video": video_path})
            
            # 处理图像
            for image in images:
                user_content.append({"type": "image", "image": image})
            
            # 处理音频
            for audio_path in audios:
                try:
                    audio_data, _ = librosa.load(audio_path, sr=16000)
                    user_content.append({"type": "audio", "audio": audio_data})
                except Exception as e:
                    logger.warning(f"Audio processing failed: {e}")
            
            if not user_content:
                user_content.append({"type": "text", "text": "Hello"})
            
            messages.append({"role": "user", "content": user_content})
            
            yield StreamingChunk(
                event="progress",
                data={"message": "准备模型输入...", "step": "input_preparation"}
            )
            
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
            
            device = next(self.model.parameters()).device
            inputs = inputs.to(device).to(self.model.dtype)
            
            input_tokens = inputs['input_ids'].shape[1]
            
            yield StreamingChunk(
                event="progress",
                data={"message": "开始流式生成...", "step": "generation_start", "input_tokens": input_tokens}
            )
            
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
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0.0 else None,
                top_p=top_p if temperature > 0.0 else None,
                do_sample=temperature > 0.0,
                use_audio_in_video=True,
                return_audio=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            
            # 在单独线程中生成
            def generate():
                with torch.no_grad():
                    self.model.generate(**generation_kwargs)
            
            thread = threading.Thread(target=generate)
            thread.start()
            
            # 流式输出
            response_text = ""
            token_count = 0
            
            for new_text in streamer:
                if new_text.strip():
                    response_text += new_text
                    token_count += 1
                    
                    current_time = time.time() - start_time
                    yield StreamingChunk(
                        event="token",
                        data={
                            "token": new_text,
                            "text": response_text,
                            "token_count": token_count,
                            "elapsed_time": current_time
                        }
                    )
            
            thread.join()
            
            # 最终结果
            processing_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None
            
            yield StreamingChunk(
                event="done",
                data={
                    "final_text": response_text,
                    "processing_time": processing_time,
                    "peak_memory_mb": peak_memory,
                    "tokens_generated": token_count,
                    "extracted_audio_url": extracted_audio_url,
                    "extracted_image_url": extracted_image_url,
                    "model_info": {
                        "model_path": self.model_path,
                        "device": str(device),
                        "dtype": str(self.model.dtype)
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Streaming processing failed: {e}")
            yield StreamingChunk(
                event="error",
                data={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "processing_time": time.time() - start_time
                }
            )

    def cleanup_temp_files(self):
        """清理临时文件"""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {file_path}: {e}")
        self.temp_files.clear()


# 全局模型管理器
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    logger.info("🚀 Starting Qwen2.5-Omni Advanced Multimodal API...")
    success = await model_manager.load_model_async()
    if success:
        logger.info("✅ API service started successfully")
    else:
        logger.error("❌ Failed to start API service")


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理资源"""
    model_manager.cleanup_temp_files()
    logger.info("👋 API service shutdown")


async def save_uploaded_files(images: List[UploadFile], audios: List[UploadFile], videos: List[UploadFile]):
    """保存上传的文件"""
    saved_images = []
    saved_audios = []
    saved_videos = []
    
    # 保存图像
    if images:
        for img_file in images:
            image = Image.open(img_file.file)
            saved_images.append(image)
    
    # 保存音频
    if audios:
        for audio_file in audios:
            temp_path = f"/tmp/audio_{int(time.time() * 1000)}_{audio_file.filename}"
            with open(temp_path, "wb") as f:
                f.write(await audio_file.read())
            saved_audios.append(temp_path)
            model_manager.temp_files.append(temp_path)
    
    # 保存视频
    if videos:
        for video_file in videos:
            temp_path = f"/tmp/video_{int(time.time() * 1000)}_{video_file.filename}"
            with open(temp_path, "wb") as f:
                f.write(await video_file.read())
            saved_videos.append(temp_path)
            model_manager.temp_files.append(temp_path)
    
    return saved_images, saved_audios, saved_videos


@app.post("/multimodal/standard", response_model=StandardResponse)
async def multimodal_standard(
    background_tasks: BackgroundTasks,
    request: StandardRequest,
    images: Optional[List[UploadFile]] = File(None),
    audios: Optional[List[UploadFile]] = File(None),  
    videos: Optional[List[UploadFile]] = File(None)
):
    """标准多模态推理接口"""
    if not model_manager.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 保存上传的文件
        saved_images, saved_audios, saved_videos = await save_uploaded_files(
            images or [], audios or [], videos or []
        )
        
        # 处理请求
        result = await model_manager.process_multimodal_standard(
            text_input=request.text,
            images=saved_images,
            audios=saved_audios,
            videos=saved_videos,
            system_prompt=request.system_prompt,
            max_new_tokens=request.max_new_tokens,
            extract_video_audio=request.extract_video_audio,
            extract_video_frame=request.extract_video_frame,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # 后台清理临时文件
        background_tasks.add_task(model_manager.cleanup_temp_files)
        
        return result
        
    except Exception as e:
        logger.error(f"Standard API request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multimodal/streaming")
async def multimodal_streaming(
    request: StreamingRequest,
    images: Optional[List[UploadFile]] = File(None),
    audios: Optional[List[UploadFile]] = File(None),  
    videos: Optional[List[UploadFile]] = File(None)
):
    """流式多模态推理接口"""
    if not model_manager.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 保存上传的文件
        saved_images, saved_audios, saved_videos = await save_uploaded_files(
            images or [], audios or [], videos or []
        )
        
        async def generate_stream():
            try:
                async for chunk in model_manager.process_multimodal_streaming(
                    text_input=request.text,
                    images=saved_images,
                    audios=saved_audios,
                    videos=saved_videos,
                    system_prompt=request.system_prompt,
                    max_new_tokens=request.max_new_tokens,
                    extract_video_audio=request.extract_video_audio,
                    extract_video_frame=request.extract_video_frame,
                    temperature=request.temperature,
                    top_p=request.top_p
                ):
                    yield f"data: {chunk.json()}\n\n"
                
                # 清理临时文件
                model_manager.cleanup_temp_files()
                
            except Exception as e:
                error_chunk = StreamingChunk(
                    event="error",
                    data={"error": str(e)}
                )
                yield f"data: {error_chunk.json()}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming API request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """提供音频文件访问"""
    file_path = f"/tmp/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return StreamingResponse(
        open(file_path, "rb"),
        media_type="audio/wav",
        headers={"Content-Disposition": f"inline; filename={filename}"}
    )


@app.get("/image/{filename}")
async def serve_image(filename: str):
    """提供图像文件访问"""
    file_path = f"/tmp/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    return StreamingResponse(
        open(file_path, "rb"),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename={filename}"}
    )


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy", 
        "model_loaded": model_manager.model_loaded,
        "device": model_manager.device,
        "model_path": model_manager.model_path
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Qwen2.5-Omni 高级多模态API服务",
        "version": "2.0.0",
        "features": [
            "标准多模态推理",
            "流式响应输出",
            "视频音轨提取",
            "视频帧提取",
            "多种输入模态支持"
        ],
        "endpoints": {
            "standard": "/multimodal/standard - 标准多模态推理",
            "streaming": "/multimodal/streaming - 流式多模态推理",
            "health": "/health - 健康检查",
            "audio": "/audio/{filename} - 音频文件访问",
            "image": "/image/{filename} - 图像文件访问"
        },
        "docs": "/docs - API文档"
    }


if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
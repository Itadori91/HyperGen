"""OpenAI-compatible API server for HyperGen."""

from __future__ import annotations

import asyncio
import base64
import io
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from hypergen.core.model import Model


class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible image generation request."""
    
    model: str
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    quality: str = "standard"
    style: str = "vivid"
    response_format: str = "url"
    user: Optional[str] = None
    
    
class ImageGenerationResponse(BaseModel):
    """OpenAI-compatible image generation response."""
    
    created: int
    data: List[Dict[str, str]]
    

class CompletionRequest(BaseModel):
    """Text completion request for prompts."""
    
    model: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    n: int = 1
    stream: bool = False
    

class ModelInfo(BaseModel):
    """Model information."""
    
    id: str
    object: str = "model"
    created: int
    owned_by: str = "hypergen"
    

class ModelsResponse(BaseModel):
    """Models list response."""
    
    object: str = "list"
    data: List[ModelInfo]
    

class QueueManager:
    """Request queue manager."""
    
    def __init__(self, max_concurrent: int = 1) -> None:
        """Initialize queue manager.
        
        Args:
            max_concurrent: Maximum concurrent requests
        """
        self.queue: asyncio.Queue = asyncio.Queue()
        self.max_concurrent = max_concurrent
        self.active_tasks = 0
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
    async def add_request(
        self,
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Add request to queue.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((func, args, kwargs, future))
        return await future
        
    async def process_queue(self) -> None:
        """Process queued requests."""
        while True:
            if self.active_tasks < self.max_concurrent:
                try:
                    func, args, kwargs, future = await asyncio.wait_for(
                        self.queue.get(), timeout=0.1
                    )
                    self.active_tasks += 1
                    
                    try:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            self.executor, func, *args, **kwargs
                        )
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                    finally:
                        self.active_tasks -= 1
                        
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.1)
                    
                    
class HyperGenServer:
    """OpenAI-compatible server for HyperGen."""
    
    def __init__(
        self,
        model: Optional[Model] = None,
        model_id: Optional[str] = None,
        max_concurrent_requests: int = 1,
        enable_optimizations: bool = True,
    ) -> None:
        """Initialize server.
        
        Args:
            model: Pre-loaded model
            model_id: Model ID to load
            max_concurrent_requests: Maximum concurrent requests
            enable_optimizations: Enable automatic optimizations
        """
        self.app = FastAPI(title="HyperGen API", version="0.1.0")
        self.setup_middleware()
        self.setup_routes()
        
        if model is None and model_id is not None:
            self.model = Model.load(
                model_id,
                enable_cpu_offload=enable_optimizations,
                compile_mode="regional" if enable_optimizations else "none",
            )
        else:
            self.model = model
            
        self.queue_manager = QueueManager(max_concurrent_requests)
        self.models_info = self._get_models_info()
        
        asyncio.create_task(self.queue_manager.process_queue())
        
    def setup_middleware(self) -> None:
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def setup_routes(self) -> None:
        """Setup API routes."""
        self.app.post("/v1/images/generations")(self.create_image)
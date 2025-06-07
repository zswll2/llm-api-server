import time
import asyncio
import os
import threading
from typing import Dict, Optional, Any, List
from vllm import LLM
from core.config import settings
from core.logger import logger

class ModelManager:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def initialize(cls):
        return cls.get_instance()
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_last_used: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._unload_task = None
        
        # 启动后台任务检查未使用模型
        if settings.models.unload_timeout > 0:
            self._start_unload_checker()
    
    def _start_unload_checker(self):
        """启动后台任务检查未使用模型"""
        async def _check_unused_models():
            while True:
                await asyncio.sleep(60)  # 每分钟检查一次
                await self._check_unused_models()
        
        self._unload_task = asyncio.create_task(_check_unused_models())
    
    async def _check_unused_models(self):
        """检查并卸载长时间未使用的模型"""
        async with self._lock:
            current_time = time.time()
            timeout = settings.models.unload_timeout
            
            models_to_unload = []
            for model_id, last_used in list(self.model_last_used.items()):
                # 跳过预加载的模型
                if model_id in settings.models.preload:
                    continue
                    
                # 如果模型超过指定时间未使用，卸载它
                if current_time - last_used > timeout:
                    models_to_unload.append(model_id)
            
            for model_id in models_to_unload:
                logger.info(f"Model {model_id} unused for {timeout}s, unloading")
                await self.unload_model(model_id)
    
    async def load_model(self, model_id: str) -> LLM:
        """加载指定ID的模型"""
        async with self._lock:
            if model_id not in settings.models.available:
                raise ValueError(f"Model {model_id} not available in configuration")
            
            # 如果模型已加载，更新使用时间并返回
            if model_id in self.models:
                self.model_last_used[model_id] = time.time()
                return self.models[model_id]["llm"]
            
            # 检查是否达到最大加载模型数量
            if len(self.models) >= settings.models.max_loaded:
                # 尝试卸载最久未使用的模型
                await self._unload_least_recently_used()
            
            # 获取模型配置
            model_config = settings.models.available[model_id]
            
            # 加载模型
            logger.info(f"Loading model: {model_id} ({model_config.path})")
            try:
                llm = LLM(
                    model=model_config.path,
                    tensor_parallel_size=model_config.tensor_parallel_size,
                    gpu_memory_utilization=model_config.gpu_memory_utilization,
                    quantization=model_config.quantization,
                    trust_remote_code=model_config.trust_remote_code,
                    max_model_len=model_config.max_tokens,
                    download_dir=os.path.abspath(settings.models.models_dir),
                )
                
                # 存储模型信息
                self.models[model_id] = {
                    "llm": llm,
                    "config": model_config
                }
                self.model_last_used[model_id] = time.time()
                
                logger.info(f"Model {model_id} loaded successfully")
                return llm
                
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {str(e)}")
                raise
    
    async def get_model(self, model_id: Optional[str] = None) -> LLM:
        """获取模型，如果未指定则使用默认模型"""
        model_id = model_id or settings.models.default
        
        # 检查模型是否已加载
        if model_id in self.models:
            self.model_last_used[model_id] = time.time()
            return self.models[model_id]["llm"]
        
        # 加载模型
        return await self.load_model(model_id)
    
    async def unload_model(self, model_id: str) -> bool:
        """卸载指定模型"""
        async with self._lock:
            if model_id not in self.models:
                return False
            
            logger.info(f"Unloading model: {model_id}")
            
            try:
                # 显式释放资源
                llm = self.models[model_id]["llm"]
                if hasattr(llm, 'llm_engine'):
                    llm.llm_engine.shutdown()
                elif hasattr(llm, 'engine'):
                    llm.engine.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down model engine: {str(e)}")
            
            # 删除模型引用
            del self.models[model_id]
            if model_id in self.model_last_used:
                del self.model_last_used[model_id]
            
            return True
    
    async def unload_all_models(self) -> None:
        """卸载所有模型"""
        if self._unload_task:
            self._unload_task.cancel()
            try:
                await self._unload_task
            except asyncio.CancelledError:
                pass
        
        model_ids = list(self.models.keys())
        for model_id in model_ids:
            await self.unload_model(model_id)
    
    async def _unload_least_recently_used(self) -> bool:
        """卸载最久未使用的模型"""
        if not self.model_last_used:
            return False
        
        # 找出最久未使用的模型
        lru_model = min(self.model_last_used.items(), key=lambda x: x[1])
        model_id, last_used = lru_model
        
        # 不卸载预加载的模型，除非必须
        if model_id in settings.models.preload and len(self.models) <= len(settings.models.preload):
            return False
        
        return await self.unload_model(model_id)
    
    def get_loaded_models(self) -> List[str]:
        """获取已加载的模型列表"""
        return list(self.models.keys())

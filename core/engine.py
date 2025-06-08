from typing import Optional, List, Dict, Any, Union, Tuple
from enum import auto, Enum
import torch
import time
import uuid
import threading
import asyncio
from vllm import LLM, SamplingParams
import logging

# 定义模型类型枚举
class ModelType(Enum):
    LLAMA = "llama"
    MISTRAL = "mistral"
    DEFAULT = "default"

# 定义引擎状态枚举
class EngineStatus(Enum):
    IDLE = auto()
    BUSY = auto()
    ERROR = auto()

class BaseEngine:
    def __init__(self, *args, **kwargs):
        self.model_path = kwargs.get('model_path')
        self.tokenizer_path = kwargs.get('tokenizer_path')
        self.max_model_len = kwargs.get('max_model_len')
        self.dtype = kwargs.get('dtype', 'auto')
        self.load_format = kwargs.get('load_format', None)
        self.gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.9)
        self.trust_remote_code = kwargs.get('trust_remote_code', True)
        self.tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
        self.disable_log_stats = kwargs.get('disable_log_stats', True)
        self.disable_log_requests = kwargs.get('disable_log_requests', True)
        self.revision = kwargs.get('revision', None)
        
        # 存储引擎实例
        self.engine = None
        # 引擎状态
        self.status = EngineStatus.IDLE
        # 存储请求和响应的映射
        self.requests = {}
        # 线程锁，用于同步访问
        self.lock = threading.Lock()
        # 会话ID到请求ID的映射
        self.conversation_map = {}

    def get_intermediate_results(self, engine):
        """获取中间生成结果的方法，修复 'scheduler' 属性问题"""
        try:
            # 在 vLLM 0.9.0.1 中，直接使用 engine._request_tracker 获取正在处理的请求
            # 而不是通过 engine.scheduler.running 访问
            request_ids = list(engine._request_tracker.get_unfinished_requests())
            if not request_ids:
                return None
                
            # 获取第一个正在处理的请求ID
            request_id = request_ids[0]
            # 从请求跟踪器中获取中间结果
            intermediate_results = engine._request_tracker.get_intermediate_results(request_id)
            return intermediate_results
        except Exception as e:
            logging.debug(f"Error getting intermediate results: {e}")
            return None

    def load_model(self):
        """加载模型"""
        try:
            self.status = EngineStatus.BUSY
            self.engine = LLM(
                model=self.model_path,
                tokenizer=self.tokenizer_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                max_model_len=self.max_model_len,
                load_format=self.load_format,
                disable_log_stats=self.disable_log_stats,
                disable_log_requests=self.disable_log_requests,
                revision=self.revision
            )
            self.status = EngineStatus.IDLE
            return True
        except Exception as e:
            self.status = EngineStatus.ERROR
            logging.error(f"Error loading model: {e}")
            return False

    def unload_model(self):
        """卸载模型"""
        if self.engine is None:
            return True
            
        try:
            self.status = EngineStatus.BUSY
            # 释放模型资源
            if hasattr(self.engine, 'llm_engine'):
                self.engine.llm_engine.shutdown()
            elif hasattr(self.engine, 'engine'):
                self.engine.engine.shutdown()
            
            self.engine = None
            self.status = EngineStatus.IDLE
            return True
        except Exception as e:
            self.status = EngineStatus.ERROR
            logging.error(f"Error unloading model: {e}")
            return False

    def generate(self, prompt, sampling_params):
        """生成文本"""
        if self.engine is None or self.status != EngineStatus.IDLE:
            raise RuntimeError("Engine not ready")
            
        try:
            self.status = EngineStatus.BUSY
            outputs = self.engine.generate(prompt, sampling_params)
            self.status = EngineStatus.IDLE
            return outputs
        except Exception as e:
            self.status = EngineStatus.ERROR
            logging.error(f"Error generating text: {e}")
            raise

    async def generate_async(self, prompt, sampling_params):
        """异步生成文本"""
        if self.engine is None or self.status != EngineStatus.IDLE:
            raise RuntimeError("Engine not ready")
            
        try:
            self.status = EngineStatus.BUSY
            # 使用asyncio运行同步函数
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None, 
                lambda: self.engine.generate(prompt, sampling_params)
            )
            self.status = EngineStatus.IDLE
            return outputs
        except Exception as e:
            self.status = EngineStatus.ERROR
            logging.error(f"Error generating text: {e}")
            raise

    async def generate_stream(self, prompt, sampling_params, callback=None):
        """异步流式生成文本"""
        if self.engine is None or self.status != EngineStatus.IDLE:
            raise RuntimeError("Engine not ready")
            
        conversation_id = str(uuid.uuid4())
        
        try:
            self.status = EngineStatus.BUSY
            
            # 启动生成任务
            loop = asyncio.get_event_loop()
            generation_task = loop.run_in_executor(
                None, 
                lambda: self.engine.generate([prompt], sampling_params)
            )
            
            last_text = ""
            finished = False
            
            # 循环检查生成进度
            while not finished:
                # 检查任务是否完成
                if generation_task.done():
                    outputs = generation_task.result()
                    output_text = outputs[0].outputs[0].text
                    finish_reason = outputs[0].outputs[0].finish_reason or "stop"
                    
                    # 发送最后一部分文本（如果有）
                    if output_text != last_text:
                        delta_text = output_text[len(last_text):]
                        if callback:
                            await callback(delta_text, conversation_id, False, False)
                    
                    # 发送完成信号
                    if callback:
                        await callback("", conversation_id, False, True, finish_reason)
                        
                    finished = True
                    
                else:
                    # 尝试获取中间结果
                    try:
                        # 这里需要访问vLLM内部状态，可能会因vLLM版本不同而需要调整
                        if hasattr(self.engine, 'llm_engine'):
                            engine = self.engine.llm_engine
                        elif hasattr(self.engine, 'engine'):
                            engine = self.engine.engine
                        else:
                            # 如果无法获取引擎，等待一小段时间后继续
                            await asyncio.sleep(0.1)
                            continue
                        
                        # 尝试获取当前生成的文本
                        intermediate_results = self.get_intermediate_results(engine)
                        if intermediate_results:
                            output_text = intermediate_results.outputs[0].text
                            
                            # 只发送增量部分
                            if output_text != last_text:
                                delta_text = output_text[len(last_text):]
                                last_text = output_text
                                
                                if delta_text and callback:
                                    await callback(delta_text, conversation_id, False if last_text else True, False)
                    except Exception as e:
                        # 忽略获取中间结果的错误，继续等待完成
                        logging.debug(f"Error getting intermediate results: {e}")
                    
                    # 短暂等待后继续检查
                    await asyncio.sleep(0.1)
            
            self.status = EngineStatus.IDLE
            return conversation_id
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            logging.error(f"Error in stream generation: {e}")
            if callback:
                await callback(str(e), conversation_id, False, True, "error")
            raise

    def is_ready(self):
        """检查引擎是否准备好"""
        return self.engine is not None and self.status == EngineStatus.IDLE

    def get_status(self):
        """获取引擎状态"""
        return {
            "status": self.status.name,
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "is_ready": self.is_ready()
        }

    def get_model_info(self):
        """获取模型信息"""
        if not self.is_ready():
            return None
            
        try:
            info = {
                "model_path": self.model_path,
                "tokenizer_path": self.tokenizer_path,
                "max_model_len": self.max_model_len,
                "dtype": self.dtype,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "tensor_parallel_size": self.tensor_parallel_size
            }
            
            # 尝试获取更多模型信息
            if hasattr(self.engine, 'llm_engine') and hasattr(self.engine.llm_engine, 'model_config'):
                model_config = self.engine.llm_engine.model_config
                info.update({
                    "model_type": model_config.model_type,
                    "vocab_size": model_config.vocab_size,
                    "hidden_size": model_config.hidden_size,
                    "num_hidden_layers": model_config.num_hidden_layers,
                    "num_attention_heads": model_config.num_attention_heads
                })
                
            return info
        except Exception as e:
            logging.error(f"Error getting model info: {e}")
            return None

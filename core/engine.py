from typing import Optional, List, Dict, Any, Union, Tuple
from enum import auto, Enum
import torch
import time
from vllm import LLM, SamplingParams
import logging

# 更多原有代码...

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

# 其余的原有代码...

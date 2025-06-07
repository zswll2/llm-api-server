import time
import json
import asyncio
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel

from core.model_manager import ModelManager
from core.config import settings
from core.logger import logger
from api.auth import get_api_key
from api.models import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamResponse,
    ModelListResponse,
    ModelObject
)
from vllm import SamplingParams
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

router = APIRouter(tags=["OpenAI Compatible API"])

# 获取可用模型列表
@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(get_api_key)):
    models = []
    for model_id, model_config in settings.models.available.items():
        models.append(
            ModelObject(
                id=model_id,
                object="model",
                created=int(time.time()),
                owned_by="organization",
                permission=[],
                root=model_id,
                parent=None
            )
        )
    return ModelListResponse(data=models, object="list")

# Chat Completions API
@router.post("/v1/chat/completions", response_model=Union[ChatCompletionResponse, ChatCompletionStreamResponse])
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(get_api_key)
):
    model_id = request.model
    stream = request.stream
    
    # 获取模型管理器
    model_manager = ModelManager.get_instance()
    
    try:
        # 加载模型（如果需要）
        llm = await model_manager.get_model(model_id)
        
        # 构建提示
        messages = request.messages
        prompt = await _build_prompt(messages, model_id)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or settings.inference.max_total_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            stop=request.stop,
        )
        
        # 优化首字延迟
        if settings.inference.optimize_first_token:
            sampling_params.best_of = 1
            sampling_params.use_beam_search = False
        
        # 流式响应
        if stream:
            return StreamingResponse(
                _generate_stream_fixed(llm, prompt, sampling_params, model_id),
                media_type="text/event-stream"
            )
        
        # 非流式响应
        outputs = await _generate_completion(llm, prompt, sampling_params)
        completion = outputs[0].outputs[0].text
        
        # 构建响应
        response = ChatCompletionResponse(
            id=f"chatcmpl-{_generate_id()}",
            object="chat.completion",
            created=int(time.time()),
            model=model_id,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion
                },
                "finish_reason": outputs[0].outputs[0].finish_reason or "stop"
            }],
            usage={
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(outputs[0].outputs[0].token_ids),
                "total_tokens": len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids)
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Text Completions API (Legacy)
@router.post("/v1/completions", response_model=Union[CompletionResponse, CompletionStreamResponse])
async def create_completion(
    request: CompletionRequest,
    api_key: str = Depends(get_api_key)
):
    model_id = request.model
    stream = request.stream
    
    # 获取模型管理器
    model_manager = ModelManager.get_instance()
    
    try:
        # 加载模型（如果需要）
        llm = await model_manager.get_model(model_id)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or settings.inference.max_total_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            stop=request.stop,
        )
        
        # 优化首字延迟
        if settings.inference.optimize_first_token:
            sampling_params.best_of = 1
            sampling_params.use_beam_search = False
        
        # 流式响应
        if stream:
            return StreamingResponse(
                _generate_completion_stream_fixed(llm, request.prompt, sampling_params, model_id),
                media_type="text/event-stream"
            )
        
        # 非流式响应
        outputs = await _generate_completion(llm, request.prompt, sampling_params)
        completion = outputs[0].outputs[0].text
        
        # 构建响应
        response = CompletionResponse(
            id=f"cmpl-{_generate_id()}",
            object="text_completion",
            created=int(time.time()),
            model=model_id,
            choices=[{
                "text": completion,
                "index": 0,
                "logprobs": None,
                "finish_reason": outputs[0].outputs[0].finish_reason or "stop"
            }],
            usage={
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(outputs[0].outputs[0].token_ids),
                "total_tokens": len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids)
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in text completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 辅助函数
def _generate_id():
    return str(uuid.uuid4())

async def _build_prompt(messages, model_id):
    """根据不同模型构建提示"""
    model_type = settings.models.available[model_id].type
    
    if model_type == "llama":
        # Llama 2 格式
        prompt = ""
        system_prompt = ""
        is_first = True
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                if is_first and system_prompt:
                    prompt += f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{content} [/INST]"
                    is_first = False
                elif is_first:
                    prompt += f"<s>[INST] {content} [/INST]"
                    is_first = False
                else:
                    prompt += f"</s><s>[INST] {content} [/INST]"
            elif role == "assistant":
                prompt += f" {content}"
        
        return prompt
        
    elif model_type == "mistral":
        # Mistral 格式
        prompt = ""
        is_first = True
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                if is_first:
                    prompt += f"<s>[INST] {content} "
                    is_first = False
                else:
                    prompt += f"{content} "
            elif role == "user":
                if is_first:
                    prompt += f"<s>[INST] {content} [/INST]"
                    is_first = False
                else:
                    prompt += f"</s><s>[INST] {content} [/INST]"
            elif role == "assistant":
                prompt += f" {content}"
        
        return prompt
        
    else:
        # 默认格式（简单拼接）
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            prompt += f"{role}: {content}\n"
        prompt += "assistant: "
        return prompt

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception)
)
async def _generate_completion(llm, prompt, sampling_params):
    """生成完成（带重试）"""
    try:
        # 使用asyncio运行同步函数
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None, 
            lambda: llm.generate([prompt], sampling_params)
        )
        return outputs
    except Exception as e:
        logger.error(f"Error in _generate_completion: {str(e)}")
        raise

# 修复版本的流式生成函数，不使用generate_iterator
async def _generate_stream_fixed(llm, prompt, sampling_params, model_id):
    """生成流式响应（修复版本）"""
    # 创建响应ID
    response_id = f"chatcmpl-{_generate_id()}"
    created = int(time.time())
    
    try:
        # 发送开始标记
        start_response = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(start_response)}\n\n"
        
        # 启动生成
        loop = asyncio.get_event_loop()
        generation_task = loop.run_in_executor(
            None, 
            lambda: llm.generate([prompt], sampling_params)
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
                    
                    delta_response = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta_text},
                            "finish_reason": None
                        }]
                    }
                    
                    yield f"data: {json.dumps(delta_response)}\n\n"
                
                # 发送完成标记
                final_response = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason
                    }]
                }
                
                yield f"data: {json.dumps(final_response)}\n\n"
                finished = True
                
            else:
                # 尝试获取中间结果
                try:
                    # 这里需要访问vLLM内部状态，可能会因vLLM版本不同而需要调整
                    if hasattr(llm, 'llm_engine'):
                        engine = llm.llm_engine
                    elif hasattr(llm, 'engine'):
                        engine = llm.engine
                    else:
                        # 如果无法获取引擎，等待一小段时间后继续
                        await asyncio.sleep(0.1)
                        continue
                    
                    # 尝试获取当前生成的文本
                    seq_group_id = list(engine.scheduler.running.keys())[0] if engine.scheduler.running else None
                    if seq_group_id:
                        sequence = engine.scheduler.running[seq_group_id].sequences[0]
                        output_text = engine.detokenizer.detokenize(sequence.output_ids)
                        
                        # 只发送增量部分
                        if output_text != last_text:
                            delta_text = output_text[len(last_text):]
                            last_text = output_text
                            
                            if delta_text:
                                delta_response = {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_id,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": delta_text},
                                        "finish_reason": None
                                    }]
                                }
                                
                                yield f"data: {json.dumps(delta_response)}\n\n"
                except Exception as e:
                    # 忽略获取中间结果的错误，继续等待完成
                    logger.debug(f"Error getting intermediate results: {str(e)}")
                
                # 短暂等待后继续检查
                await asyncio.sleep(0.1)
        
        # 发送结束标记
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in stream generation: {str(e)}")
        error_response = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"

# 修复版本的文本完成流式生成函数
async def _generate_completion_stream_fixed(llm, prompt, sampling_params, model_id):
    """生成文本完成流式响应（修复版本）"""
    # 创建响应ID
    response_id = f"cmpl-{_generate_id()}"
    created = int(time.time())
    
    try:
        # 启动生成
        loop = asyncio.get_event_loop()
        generation_task = loop.run_in_executor(
            None, 
            lambda: llm.generate([prompt], sampling_params)
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
                    
                    stream_response = {
                        "id": response_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "text": delta_text,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None
                        }]
                    }
                    
                    yield f"data: {json.dumps(stream_response)}\n\n"
                
                # 发送完成标记
                final_response = {
                    "id": response_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason
                    }]
                }
                
                yield f"data: {json.dumps(final_response)}\n\n"
                finished = True
                
            else:
                # 尝试获取中间结果
                try:
                    # 这里需要访问vLLM内部状态，可能会因vLLM版本不同而需要调整
                    if hasattr(llm, 'llm_engine'):
                        engine = llm.llm_engine
                    elif hasattr(llm, 'engine'):
                        engine = llm.engine
                    else:
                        # 如果无法获取引擎，等待一小段时间后继续
                        await asyncio.sleep(0.1)
                        continue
                    
                    # 尝试获取当前生成的文本
                    seq_group_id = list(engine.scheduler.running.keys())[0] if engine.scheduler.running else None
                    if seq_group_id:
                        sequence = engine.scheduler.running[seq_group_id].sequences[0]
                        output_text = engine.detokenizer.detokenize(sequence.output_ids)
                        
                        # 只发送增量部分
                        if output_text != last_text:
                            delta_text = output_text[len(last_text):]
                            last_text = output_text
                            
                            if delta_text:
                                stream_response = {
                                    "id": response_id,
                                    "object": "text_completion",
                                    "created": created,
                                    "model": model_id,
                                    "choices": [{
                                        "text": delta_text,
                                        "index": 0,
                                        "logprobs": None,
                                        "finish_reason": None
                                    }]
                                }
                                
                                yield f"data: {json.dumps(stream_response)}\n\n"
                except Exception as e:
                    # 忽略获取中间结果的错误，继续等待完成
                    logger.debug(f"Error getting intermediate results: {str(e)}")
                
                # 短暂等待后继续检查
                await asyncio.sleep(0.1)
        
        # 发送结束标记
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in completion stream: {str(e)}")
        error_response = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"

# 旧的流式生成函数（保留作为参考）
async def _generate_stream(llm, prompt, sampling_params, model_id):
    """生成流式响应"""
    # 创建响应ID
    response_id = f"chatcmpl-{_generate_id()}"
    created = int(time.time())
    
    try:
        # 启动生成
        results_generator = llm.generate_iterator(
            prompts=[prompt],
            sampling_params=sampling_params,
        )
        
        # 发送开始标记
        start_response = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(start_response)}\n\n"
        
        last_text = ""
        
        # 迭代生成的tokens
        async for result in results_generator:
            if len(result.outputs) == 0:
                continue
                
            output_text = result.outputs[0].text
            
            # 只发送增量部分
            delta_text = output_text[len(last_text):]
            last_text = output_text
            
            if not delta_text:
                continue
            
            # 构建delta响应
            delta_response = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"content": delta_text},
                    "finish_reason": None
                }]
            }
            
            yield f"data: {json.dumps(delta_response)}\n\n"
            
            # 如果是最后一个结果
            if result.finished:
                finish_reason = "stop"
                if result.outputs[0].finish_reason:
                    finish_reason = result.outputs[0].finish_reason
                    
                final_response = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason
                    }]
                }
                
                yield f"data: {json.dumps(final_response)}\n\n"
        
        # 发送结束标记
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in stream generation: {str(e)}")
        error_response = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"

async def _generate_completion_stream(llm, prompt, sampling_params, model_id):
    """生成文本完成流式响应"""
    # 创建响应ID
    response_id = f"cmpl-{_generate_id()}"
    created = int(time.time())
    
    try:
        # 启动生成
        results_generator = llm.generate_iterator(
            prompts=[prompt],
            sampling_params=sampling_params,
        )
        
        last_text = ""
        
        # 迭代生成的tokens
        async for result in results_generator:
            if len(result.outputs) == 0:
                continue
                
            output_text = result.outputs[0].text
            
            # 只发送增量部分
            delta_text = output_text[len(last_text):]
            last_text = output_text
            
            if not delta_text:
                continue
            
            # 构建响应
            stream_response = {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "text": delta_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None
                }]
            }
            
            yield f"data: {json.dumps(stream_response)}\n\n"
            
            # 如果是最后一个结果
            if result.finished:
                finish_reason = "stop"
                if result.outputs[0].finish_reason:
                    finish_reason = result.outputs[0].finish_reason
                    
                final_response = {
                    "id": response_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason
                    }]
                }
                
                yield f"data: {json.dumps(final_response)}\n\n"
        
        # 发送结束标记
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in completion stream: {str(e)}")
        error_response = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any

# 模型相关
class ModelPermission(BaseModel):
    id: str = Field(default="")
    object: str = Field(default="model_permission")
    created: int = Field(default=0)
    allow_create_engine: bool = Field(default=False)
    allow_sampling: bool = Field(default=True)
    allow_logprobs: bool = Field(default=True)
    allow_search_indices: bool = Field(default=False)
    allow_view: bool = Field(default=True)
    allow_fine_tuning: bool = Field(default=False)
    organization: str = Field(default="*")
    group: Optional[str] = None
    is_blocking: bool = Field(default=False)

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "organization"
    permission: List[ModelPermission] = Field(default_factory=list)
    root: str
    parent: Optional[str] = None

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelObject]

# Chat Completion 相关
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: Optional[str] = None

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: Dict[str, str]
    finish_reason: Optional[str] = None

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionResponseStreamChoice]

# Text Completion 相关
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    suffix: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: int = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class CompletionResponseChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo

class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionResponseChoice]

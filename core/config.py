import os
import yaml
from pydantic import BaseModel, field_validator
from typing import Dict, List, Optional, Any

# 加载配置文件
def load_yaml_config(file_path: str = "config.yaml") -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

# 服务器配置
class ServerConfig(BaseModel):
    host: str
    port: int
    workers: int
    timeout: int

# 认证配置
class ApiKey(BaseModel):
    key: str
    name: str
    permissions: List[str]

class AuthConfig(BaseModel):
    enabled: bool
    secret_key: str
    algorithm: str
    api_keys: List[ApiKey]

# 日志配置
class LoggingConfig(BaseModel):
    level: str
    format: str
    retention: str
    rotation: str
    path: str
    console: bool
    @field_validator('path')
    def make_path_absolute(cls, v):
        return os.path.abspath(v)


# 模型配置
class ModelConfig(BaseModel):
    path: str
    type: str
    max_tokens: int
    quantization: Optional[str] = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = False

class ModelsConfig(BaseModel):
    default: str
    preload: List[str]
    max_loaded: int
    unload_timeout: int
    models_dir: str
    available: Dict[str, ModelConfig]
    @field_validator('models_dir')
    def make_path_absolute(cls, v):
        return os.path.abspath(v)
        
    @field_validator('default')
    def default_must_be_available(cls, v, values):
        if 'available' in values.data and v not in values.data['available']:
            raise ValueError(f"Default model '{v}' must be in available models list")
        return v
    
    @field_validator('preload')
    def preload_models_must_be_available(cls, v, values):
        if 'available' in values.data:
            for model_id in v:
                if model_id not in values.data['available']:
                    raise ValueError(f"Preload model '{model_id}' must be in available models list")
        return v

# 推理配置
class InferenceConfig(BaseModel):
    max_total_tokens: int
    max_input_tokens: int
    max_batch_size: int
    streaming_default: bool
    optimize_first_token: bool

# 全局设置
class Settings(BaseModel):
    server: ServerConfig
    auth: AuthConfig
    logging: LoggingConfig
    models: ModelsConfig
    inference: InferenceConfig

# 加载配置
try:
    config_data = load_yaml_config()
    settings = Settings.model_validate(config_data)
except Exception as e:
    print(f"Error loading configuration: {str(e)}")
    raise

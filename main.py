import uvicorn
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import settings
from core.logger import setup_logger, logger
from core.model_manager import ModelManager
from api.openai_compat import router as openai_router
from api.auth import router as auth_router

# 设置日志
setup_logger()

# 确保必要的目录存在
def ensure_directories():
    for directory in [settings.logging.path, settings.models.models_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

# 启动时预加载模型
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    logger.info("Starting LLM API Server...")
    
    # 初始化模型管理器并预热模型
    model_manager = ModelManager.get_instance()
    for model_id in settings.models.preload:
        if model_id in settings.models.available:
            logger.info(f"Preloading model: {model_id}")
            try:
                await model_manager.load_model(model_id)
                logger.info(f"Successfully preloaded model: {model_id}")
            except Exception as e:
                logger.error(f"Failed to preload model {model_id}: {str(e)}")
        else:
            logger.warning(f"Cannot preload model {model_id}: not in available models list")
    
    yield
    
    # 关闭时清理资源
    logger.info("Shutting down LLM API Server...")
    await model_manager.unload_all_models()

# 创建FastAPI应用
app = FastAPI(
    title="LLM API Server",
    description="OpenAI-compatible API server for LLMs using vLLM",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    path = request.url.path
    # 跳过健康检查端点的日志记录以减少噪音
    if path != "/health":
        logger.debug(f"Request: {request.method} {path}")
    response = await call_next(request)
    if path != "/health":
        logger.debug(f"Response: {response.status_code}")
    return response

# 注册路由
app.include_router(openai_router)
app.include_router(auth_router)

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "LLM API Server is running",
        "default_model": settings.models.default,
        "available_models": list(settings.models.available.keys()),
    }

@app.get("/health")
async def health():
    model_manager = ModelManager.get_instance()
    loaded_models = model_manager.get_loaded_models()
    
    status = "healthy"
    details = {}
    
    # 检查默认模型是否加载
    default_model = settings.models.default
    if default_model not in loaded_models:
        status = "degraded"
        details["error"] = f"Default model {default_model} not loaded"
    
    # 检查预加载模型
    for model_id in settings.models.preload:
        if model_id not in loaded_models:
            status = "degraded"
            details.setdefault("missing_models", []).append(model_id)
    
    return {
        "status": status,
        "loaded_models": loaded_models,
        "default_model": default_model,
        "preload_models": settings.models.preload,
        "details": details
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
        timeout_keep_alive=settings.server.timeout,
    )

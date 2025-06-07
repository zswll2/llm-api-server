import os
import sys
from loguru import logger
from core.config import settings

def setup_logger():
    # 清除默认处理程序
    logger.remove()
    
    # 确保日志目录存在
    log_path = settings.logging.path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        logger.info(f"Created log directory: {log_path}")
    
    # 设置日志文件
    log_file = os.path.join(log_path, "llm_api.log")
    
    # 添加文件处理程序
    logger.add(
        log_file,
        level=settings.logging.level,
        format=settings.logging.format,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        enqueue=True,  # 启用多进程安全
    )
    
    # 添加控制台处理程序
    if settings.logging.console:
        logger.add(
            sys.stderr,
            level=settings.logging.level,
            format=settings.logging.format,
            enqueue=True,  # 启用多进程安全
        )
    
    return logger

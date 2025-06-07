import time
import uuid
import json
from typing import Dict, List, Any, Optional

def generate_unique_id(prefix: str = "gen") -> str:
    """生成唯一ID"""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

def format_datetime() -> str:
    """格式化当前时间为ISO格式"""
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

def count_tokens(text: str) -> int:
    """简单的token计数估算（仅用于演示）"""
    # 实际应用中应使用模型的tokenizer
    return len(text.split())

def ensure_json_response(text: str) -> Dict[str, Any]:
    """确保返回有效的JSON对象"""
    try:
        # 尝试直接解析
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试从文本中提取JSON
        try:
            # 查找JSON代码块
            if "```json" in text:
                json_text = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_text)
            elif "```" in text:
                json_text = text.split("```")[1].split("```")[0].strip()
                return json.loads(json_text)
        except (IndexError, json.JSONDecodeError):
            pass
        
        # 返回原始文本
        return {"content": text}

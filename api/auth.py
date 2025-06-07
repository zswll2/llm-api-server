from fastapi import APIRouter, Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel

from core.config import settings
from core.logger import logger

router = APIRouter(tags=["Authentication"], prefix="/auth")

security = HTTPBearer(auto_error=False)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    permissions: List[str] = []

def get_api_key(
    authorization: Optional[str] = Header(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """验证API密钥"""
    if not settings.auth.enabled:
        return "disabled"
        
    # 获取API密钥
    api_key = None
    
    # 从Authorization头获取
    if authorization:
        if authorization.startswith("Bearer "):
            api_key = authorization.replace("Bearer ", "")
        elif authorization.startswith("sk-"):
            api_key = authorization
    
    # 从安全依赖获取
    if credentials:
        api_key = credentials.credentials
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 验证API密钥
    for key_config in settings.auth.api_keys:
        if key_config.key == api_key:
            return api_key
    
    # API密钥无效
    logger.warning(f"Invalid API key attempt: {api_key[:5]}...")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建JWT访问令牌"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.auth.secret_key, 
        algorithm=settings.auth.algorithm
    )
    
    return encoded_jwt

@router.post("/token", response_model=Token)
async def login_for_access_token(api_key: str = Depends(get_api_key)):
    """使用API密钥获取访问令牌"""
    # 查找API密钥配置
    key_config = None
    for config in settings.auth.api_keys:
        if config.key == api_key:
            key_config = config
            break
    
    if not key_config:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 创建访问令牌
    access_token_expires = timedelta(days=1)
    access_token = create_access_token(
        data={
            "sub": key_config.name,
            "permissions": key_config.permissions
        },
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

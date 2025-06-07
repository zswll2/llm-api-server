# LLM API Server

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

一个高性能、兼容OpenAI API的大语言模型服务器，支持多种开源LLM模型，基于FastAPI和vLLM构建。

## 功能特点

- 🚀 **高性能推理**：基于vLLM实现高吞吐量和低延迟的模型推理
- 🔄 **OpenAI兼容API**：完全兼容OpenAI API，可直接替代OpenAI服务
- 🔌 **多模型支持**：支持各种开源LLM，如Qwen、DeepSeek、Llama等
- 🔒 **API密钥认证**：内置API密钥管理和权限控制
- 🌊 **流式输出**：支持流式响应，减少首字延迟
- 🔄 **动态模型加载**：按需加载和卸载模型，优化资源使用
- ⚙️ **灵活配置**：通过YAML配置文件进行全面定制

## 安装指南

### 系统要求

- Python 3.10+
- CUDA 11.8+
- 至少16GB GPU内存（取决于所使用的模型）

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/zswll2/llm-api-server.git
cd llm-api-server
```

2. 创建并激活虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者
venv\Scripts\activate  # Windows
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 配置服务器

复制示例配置文件并根据需要进行修改：

```bash
cp config.example.yaml config.yaml
# 使用您喜欢的编辑器编辑config.yaml
```

5. 下载模型

下载您想要使用的模型，并确保在配置文件中正确设置模型路径。

## 配置说明

`config.yaml`文件包含了服务器的所有配置选项：

```yaml
# 服务配置
server:
  host: "0.0.0.0"  # 监听地址
  port: 8000        # 监听端口
  workers: 1        # 工作进程数
  timeout: 300      # 请求超时时间（秒）

# 认证配置
auth:
  enabled: true     # 是否启用认证
  secret_key: "your-secret-key"  # JWT密钥
  algorithm: "HS256"            # JWT算法
  api_keys:         # 预设API密钥列表
    - key: "sk-myapikey123456789"
      name: "default"
      permissions: ["all"]
    - key: "sk-restricted987654321"
      name: "restricted"
      permissions: ["chat"]

# 日志配置
logging:
  level: "INFO"     # 日志级别
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
  retention: "7 days"
  rotation: "1 day"
  path: "logs"       # 日志目录
  console: true      # 是否输出到控制台

# 模型配置
models:
  default: "DeepSeek-R1-0528-Qwen3-8B"  # 默认模型
  preload: ["DeepSeek-R1-0528-Qwen3-8B"]  # 预加载模型
  max_loaded: 2      # 最大同时加载模型数
  unload_timeout: 3600  # 模型不活跃多久后卸载（秒）
  models_dir: "models"  # 模型目录
  
  # 可用模型列表
  available:
    DeepSeek-R1-0528-Qwen3-8B:
      path: "/path/to/model"  # 模型路径
      type: "qwen"           # 模型类型
      max_tokens: 4096        # 最大token数
      quantization: null      # 量化方式
      tensor_parallel_size: 1  # 张量并行大小
      gpu_memory_utilization: 0.8  # GPU内存利用率
      trust_remote_code: true  # 是否信任远程代码

# 推理配置
inference:
  max_total_tokens: 4096  # 输入+输出的最大token数
  max_input_tokens: 3072  # 最大输入token数
  max_batch_size: 32      # 最大批处理大小
  streaming_default: true  # 默认是否流式输出
  optimize_first_token: true  # 是否优化首字延迟
```

## 启动服务器

### 开发环境

```bash
python main.py
```

### 生产环境

使用Gunicorn（Linux/Mac）：

```bash
gunicorn main:app -c gunicorn.conf.py
```

使用配置文件：

```python
# gunicorn.conf.py
chdir = '/path/to/llm-api-server'
workers = 1  # 建议与config.yaml中的workers保持一致
threads = 2
worker_class = 'uvicorn.workers.UvicornWorker'
bind = '0.0.0.0:8050'
pidfile = 'gunicorn.pid'
accesslog = 'logs/gunicorn_access.log'
errorlog = 'logs/gunicorn_error.log'
loglevel = 'info'
```

## API使用示例

服务器提供了与OpenAI兼容的API接口，可以使用任何支持OpenAI的客户端库进行调用。

### 认证

所有API请求都需要在Header中包含API密钥：

```
Authorization: Bearer sk-myapikey123456789
```

### 聊天完成（Chat Completion）

#### 基本聊天请求

```bash
curl -X POST "http://localhost:8050/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-myapikey123456789" \
  -d '{
    "model": "DeepSeek-R1-0528-Qwen3-8B",
    "messages": [
      {"role": "system", "content": "你是一个有用的AI助手。"},
      {"role": "user", "content": "介绍一下自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

#### 流式聊天请求

```bash
curl -X POST "http://localhost:8050/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-myapikey123456789" \
  -d '{
    "model": "DeepSeek-R1-0528-Qwen3-8B",
    "messages": [
      {"role": "system", "content": "你是一个有用的AI助手。"},
      {"role": "user", "content": "写一首关于春天的诗"}
    ],
    "temperature": 0.7,
    "max_tokens": 500,
    "stream": true
  }'
```

### 文本完成（Text Completion）

```bash
curl -X POST "http://localhost:8050/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-myapikey123456789" \
  -d '{
    "model": "DeepSeek-R1-0528-Qwen3-8B",
    "prompt": "讲一个关于人工智能的笑话",
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

### 模型列表

获取可用模型列表：

```bash
curl -X GET "http://localhost:8050/v1/models" \
  -H "Authorization: Bearer sk-myapikey123456789"
```

### 嵌入（Embeddings）

生成文本嵌入向量：

```bash
curl -X POST "http://localhost:8050/v1/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-myapikey123456789" \
  -d '{
    "model": "DeepSeek-R1-0528-Qwen3-8B",
    "input": "人工智能将如何改变我们的未来？"
  }'
```

## 使用Python客户端

您可以使用OpenAI的Python客户端库与服务器进行交互：

```python
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="sk-myapikey123456789",  # 您的API密钥
    base_url="http://localhost:8050/v1"  # 服务器地址
)

# 聊天完成
response = client.chat.completions.create(
    model="DeepSeek-R1-0528-Qwen3-8B",
    messages=[
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "解释一下量子计算的基本原理"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)

# 流式响应
stream = client.chat.completions.create(
    model="DeepSeek-R1-0528-Qwen3-8B",
    messages=[
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "写一篇短文，描述未来的智能城市"}
    ],
    temperature=0.7,
    max_tokens=1000,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

## 多模态模型示例

对于支持多模态功能的模型（如Qwen-VL、LLaVA等），可以通过以下方式调用：

### 多模态聊天请求

```bash
curl -X POST "http://localhost:8050/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-myapikey123456789" \
  -d '{
    "model": "Qwen-VL-Chat",
    "messages": [
      {"role": "system", "content": "你是一个有用的AI助手，能够理解图像和文本。"},
      {"role": "user", "content": [
        {"type": "text", "text": "这张图片里有什么？"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."}}  
      ]}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

### 使用Python客户端进行多模态请求

```python
from openai import OpenAI
import base64

# 初始化客户端
client = OpenAI(
    api_key="sk-myapikey123456789",
    base_url="http://localhost:8050/v1"
)

# 读取并编码图像
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 获取base64编码的图像
image_path = "path/to/your/image.jpg"
base64_image = encode_image(image_path)

# 创建多模态请求
response = client.chat.completions.create(
    model="Qwen-VL-Chat",
    messages=[
        {"role": "system", "content": "你是一个有用的AI助手，能够理解图像和文本。"},
        {"role": "user", "content": [
            {"type": "text", "text": "请描述这张图片并分析其中的主要内容。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
```

## 常见问题解答

### 1. 如何添加新模型？

在`config.yaml`的`models.available`部分添加新模型的配置：

```yaml
models:
  available:
    # 现有模型...
    新模型名称:
      path: "/path/to/new/model"
      type: "模型类型"  # 如llama, qwen, mistral等
      max_tokens: 4096
      # 其他配置...
```

### 2. 如何解决CUDA内存不足的问题？

- 减少工作进程数量（`server.workers`）
- 降低GPU内存利用率（`models.available.模型名.gpu_memory_utilization`）
- 使用模型量化（`models.available.模型名.quantization`设置为"awq"或"gptq"）
- 减少最大token数（`models.available.模型名.max_tokens`）

### 3. 如何添加新的API密钥？

在`config.yaml`的`auth.api_keys`部分添加新的API密钥：

```yaml
auth:
  api_keys:
    # 现有密钥...
    - key: "sk-新密钥"
      name: "新用户名称"
      permissions: ["chat", "completions"]  # 或["all"]
```

### 4. 流式输出报错 "'LLM' object has no attribute 'generate_iterator'" 怎么解决？

这个错误通常是因为使用的vLLM版本不支持`generate_iterator`方法。解决方法：

1. 升级vLLM到最新版本：
```bash
pip install -U vllm
```

2. 如果仍然有问题，可以修改`api/openai_compat.py`文件中的流式生成函数，使用异步迭代器手动实现流式输出。

## 性能优化

### GPU内存优化

- **工作进程数量**：确保`config.yaml`中的`server.workers`与`gunicorn.conf.py`中的`workers`保持一致
- **张量并行**：对于大模型，可以增加`tensor_parallel_size`以在多个GPU上分割模型
- **量化**：使用AWQ或GPTQ量化可以显著减少内存使用

### 推理速度优化

- **批处理**：适当增加`max_batch_size`可以提高吞吐量
- **首字优化**：保持`optimize_first_token`为`true`以减少首字延迟
- **缓存预热**：使用`preload`预加载常用模型

## 故障排除

### 模型加载失败

如果遇到模型加载失败，检查以下几点：

1. 确保GPU内存足够（使用`nvidia-smi`检查）
2. 验证模型路径是否正确
3. 检查模型文件是否完整
4. 查看日志文件获取详细错误信息

### API请求错误

如果API请求返回错误：

1. 确认API密钥格式正确（以`sk-`开头）
2. 检查请求格式是否符合OpenAI API规范
3. 验证请求的模型是否在可用模型列表中
4. 检查是否超出了token限制

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议！请通过GitHub Issues或Pull Requests参与项目开发。

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。
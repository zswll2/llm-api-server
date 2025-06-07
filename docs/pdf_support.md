# PDF文件支持指南

本文档介绍如何在llm-api-server中添加对PDF文件的支持，使模型能够处理和分析PDF文档内容。

## 实现方法

### 1. 安装必要的依赖

首先，需要安装PDF处理相关的依赖：

```bash
pip install pypdf langchain unstructured pdf2image pytesseract
```

对于OCR支持（识别扫描版PDF中的文字），还需要安装Tesseract：

```bash
# Ubuntu/Debian
apt-get install -y tesseract-ocr
# 中文支持
apt-get install -y tesseract-ocr-chi-sim
```

### 2. 创建PDF处理模块

在项目中创建一个新的模块来处理PDF文件：

```python
# core/pdf_processor.py
import os
import base64
import tempfile
from typing import List, Dict, Any, Optional

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF文件中提取文本"""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def process_base64_pdf(self, base64_pdf: str) -> List[str]:
        """处理Base64编码的PDF文件并返回分块的文本"""
        # 解码Base64字符串
        pdf_bytes = base64.b64decode(base64_pdf)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name
        
        try:
            # 提取文本
            text = self.extract_text_from_pdf(temp_path)
            
            # 分割文本为较小的块
            text_chunks = self.text_splitter.split_text(text)
            
            return text_chunks
        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def process_pdf_with_ocr(self, pdf_path: str, language: str = "eng+chi_sim") -> str:
        """使用OCR处理PDF文件（适用于扫描版PDF）"""
        from pdf2image import convert_from_path
        import pytesseract
        
        # 将PDF转换为图像
        images = convert_from_path(pdf_path)
        
        # 使用OCR提取文本
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image, lang=language) + "\n"
        
        return text
```

### 3. 扩展API以支持PDF输入

修改API模型以支持PDF输入：

```python
# api/models.py 中添加

class PDFInput(BaseModel):
    type: str = "pdf"
    pdf: str  # Base64编码的PDF

class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None
    pdf: Optional[str] = None  # Base64编码的PDF

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]

# 更新ChatCompletionRequest模型
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Union[Dict[str, Union[str, List[Dict[str, Any]]]], ChatMessage]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
```

### 4. 修改聊天完成端点以处理PDF

在`api/openai_compat.py`中修改聊天完成端点：

```python
# 在文件顶部导入PDF处理器
from core.pdf_processor import PDFProcessor

# 初始化PDF处理器
pdf_processor = PDFProcessor()

# 在_build_prompt函数中添加PDF处理逻辑
async def _build_prompt(messages, model_id):
    """根据不同模型构建提示"""
    model_type = settings.models.available[model_id].type
    
    # 处理消息中的PDF
    processed_messages = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        # 检查是否是包含PDF的复杂内容
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "pdf":
                    # 处理PDF并提取文本
                    pdf_base64 = item.get("pdf", "")
                    if pdf_base64:
                        pdf_chunks = pdf_processor.process_base64_pdf(pdf_base64)
                        text_parts.append("\n\nPDF内容:\n" + "\n\n".join(pdf_chunks))
            
            # 将处理后的文本合并
            processed_content = "\n".join(text_parts)
            processed_messages.append({"role": role, "content": processed_content})
        else:
            processed_messages.append(message)
    
    # 使用处理后的消息构建提示
    messages = processed_messages
    
    # 原有的提示构建逻辑...
```

### 5. 使用示例

以下是如何使用PDF功能的Python客户端示例：

```python
import base64
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="sk-myapikey123456789",
    base_url="http://localhost:8050/v1"
)

# 读取并编码PDF文件
def encode_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')

# 获取base64编码的PDF
pdf_path = "path/to/your/document.pdf"
base64_pdf = encode_pdf(pdf_path)

# 创建包含PDF的请求
response = client.chat.completions.create(
    model="DeepSeek-R1-0528-Qwen3-8B",
    messages=[
        {"role": "system", "content": "你是一个有用的AI助手，能够分析PDF文档并回答相关问题。"},
        {"role": "user", "content": [
            {"type": "text", "text": "请分析这份PDF文档并总结其主要内容。"},
            {"type": "pdf", "pdf": base64_pdf}
        ]}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
```

### 6. cURL示例

```bash
curl -X POST "http://localhost:8050/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-myapikey123456789" \
  -d '{
    "model": "DeepSeek-R1-0528-Qwen3-8B",
    "messages": [
      {"role": "system", "content": "你是一个有用的AI助手，能够分析PDF文档并回答相关问题。"},
      {"role": "user", "content": [
        {"type": "text", "text": "请分析这份PDF文档并总结其主要内容。"},
        {"type": "pdf", "pdf": "JVBERi0xLjMKJcTl8uXrp...（Base64编码的PDF内容）..."}
      ]}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

## 高级功能

### 1. PDF文档检索

对于长PDF文档，可以实现基于向量数据库的检索功能：

```python
# 安装必要的依赖
# pip install faiss-cpu sentence-transformers

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class PDFRetriever:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.text_chunks = []
        self.index = None
    
    def index_pdf(self, pdf_path: str):
        """为PDF创建检索索引"""
        # 提取文本并分块
        processor = PDFProcessor()
        text = processor.extract_text_from_pdf(pdf_path)
        self.text_chunks = processor.text_splitter.split_text(text)
        
        # 创建嵌入
        embeddings = self.model.encode(self.text_chunks)
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
    
    def query(self, question: str, top_k: int = 3):
        """查询与问题最相关的PDF段落"""
        question_embedding = self.model.encode([question])
        distances, indices = self.index.search(np.array(question_embedding).astype('float32'), top_k)
        
        results = []
        for idx in indices[0]:
            results.append(self.text_chunks[idx])
        
        return results
```

### 2. 表格提取

对于包含表格的PDF，可以使用专门的表格提取工具：

```python
# 安装必要的依赖
# pip install tabula-py pandas

import tabula
import pandas as pd

def extract_tables_from_pdf(pdf_path: str):
    """从PDF中提取表格"""
    # 读取所有表格
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
    
    # 转换为可读格式
    results = []
    for i, table in enumerate(tables):
        # 转换为Markdown格式
        markdown_table = table.to_markdown()
        results.append(f"表格 {i+1}:\n{markdown_table}\n")
    
    return "\n".join(results)
```

### 3. 图像识别与处理

对于包含图像的PDF，可以提取并分析图像：

```python
# 安装必要的依赖
# pip install pdf2image pillow

from pdf2image import convert_from_path
from PIL import Image
import io
import base64

def extract_images_from_pdf(pdf_path: str):
    """从PDF中提取图像"""
    images = convert_from_path(pdf_path)
    
    # 将图像转换为Base64编码
    image_data = []
    for i, img in enumerate(images):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_data.append({
            "index": i,
            "base64": img_str
        })
    
    return image_data
```

## 注意事项

1. **内存使用**：处理大型PDF文件可能会消耗大量内存，特别是在提取图像或进行OCR时。

2. **安全性**：确保对上传的PDF进行安全检查，以防止恶意PDF文件。

3. **超时处理**：大型PDF处理可能需要较长时间，确保设置适当的超时时间。

4. **错误处理**：添加健壮的错误处理机制，以应对各种PDF格式和内容。

5. **文本质量**：某些PDF（特别是扫描版）的文本提取质量可能不高，可能需要OCR或人工校正。

## 结论

通过以上步骤，您可以为llm-api-server添加PDF处理功能，使模型能够理解和分析PDF文档内容。根据您的具体需求，可以进一步扩展这些功能，如添加更高级的文档理解、表格分析或图像识别能力。
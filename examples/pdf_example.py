import base64
import time
import requests
import json
from openai import OpenAI

# 服务器配置
API_KEY = "sk-myapikey123456789"
BASE_URL = "http://localhost:8050/v1"

def encode_pdf(pdf_path):
    """读取并编码PDF文件为base64"""
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')

def process_pdf_direct(pdf_path):
    """使用PDF处理API直接处理PDF"""
    base64_pdf = encode_pdf(pdf_path)
    
    # 调用PDF处理API
    response = requests.post(
        f"{BASE_URL}/pdf/process",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        json={
            "pdf_base64": base64_pdf,
            "extract_tables": True,
            "extract_images": True
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print("PDF处理成功!")
        print(f"处理时间: {result['processing_time']:.2f}秒")
        print(f"提取的文本块数: {len(result['result']['text_chunks'])}")
        
        # 打印部分文本内容
        if result['result']['text_chunks']:
            print("\n文本样例:")
            print(result['result']['text_chunks'][0][:200] + "...")
        
        # 打印表格信息
        if result['result'].get('has_tables'):
            print("\n检测到表格:")
            print(result['result'].get('tables', [])[0][:200] + "...")
        
        # 打印图像信息
        if result['result'].get('has_images'):
            print("\n检测到图像:")
            images = result['result'].get('images', [])
            print(f"图像数量: {len(images)}")
            for i, img in enumerate(images):
                print(f"图像 {i+1}: {img['width']}x{img['height']} (页码: {img['page']})")
        
        return result
    else:
        print(f"处理失败: {response.status_code}")
        print(response.text)
        return None

def chat_with_pdf_openai_client(pdf_path, question):
    """使用OpenAI客户端通过聊天API处理PDF"""
    # 初始化客户端
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    
    # 读取并编码PDF
    base64_pdf = encode_pdf(pdf_path)
    
    # 创建包含PDF的请求
    start_time = time.time()
    
    response = client.chat.completions.create(
        model="DeepSeek-R1-0528-Qwen3-8B",  # 使用配置的模型
        messages=[
            {"role": "system", "content": "你是一个有用的AI助手，能够分析PDF文档并回答相关问题。"},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "pdf", "pdf": base64_pdf}
            ]}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n回答生成时间: {elapsed_time:.2f}秒")
    print("\nAI回答:")
    print(response.choices[0].message.content)
    
    return response.choices[0].message.content

def chat_with_pdf_curl(pdf_path, question):
    """使用curl风格的请求通过聊天API处理PDF"""
    # 读取并编码PDF
    base64_pdf = encode_pdf(pdf_path)
    
    # 构建请求
    request_data = {
        "model": "DeepSeek-R1-0528-Qwen3-8B",
        "messages": [
            {"role": "system", "content": "你是一个有用的AI助手，能够分析PDF文档并回答相关问题。"},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "pdf", "pdf": base64_pdf}
            ]}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # 发送请求
    start_time = time.time()
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        json=request_data
    )
    
    elapsed_time = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n回答生成时间: {elapsed_time:.2f}秒")
        print("\nAI回答:")
        print(result['choices'][0]['message']['content'])
        return result
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)
        return None

# 使用示例
if __name__ == "__main__":
    # 替换为您的PDF文件路径
    pdf_path = "example.pdf"
    
    print("===== 直接处理PDF =====")
    pdf_result = process_pdf_direct(pdf_path)
    
    print("\n\n===== 使用OpenAI客户端与PDF聊天 =====")
    chat_result = chat_with_pdf_openai_client(
        pdf_path, 
        "请分析这份PDF文档并总结其主要内容。如果有表格，请解释表格的含义。"
    )
    
    print("\n\n===== 使用curl风格请求与PDF聊天 =====")
    curl_result = chat_with_pdf_curl(
        pdf_path,
        "这份PDF文档的主要结论是什么？请提取文档中的关键信息和数据。"
    )
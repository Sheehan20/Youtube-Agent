#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 测试 Grok API 连接

import os
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

# 读取配置
api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('model')
base_url = os.getenv('BASE_URL')

print(f"API Key: {api_key[:20]}..." if api_key else "API Key: None")
print(f"Model: {model}")
print(f"Base URL: {base_url}")

# 测试连接
try:
    llm_params = {
        "api_key": api_key,
        "model": model,
        "temperature": 0
    }
    
    if base_url:
        llm_params["base_url"] = base_url
    
    llm = ChatOpenAI(**llm_params)
    print("\nTesting API call...")
    response = llm.invoke("Hello")
    print("API call successful!")
    print(f"Response: {response.content[:100]}...")
except Exception as e:
    print(f"\nAPI call failed!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    import traceback
    traceback.print_exc()



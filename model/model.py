import os

import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "你是运维小助手"},
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "thinking": {"type": "enabled"},
    "reasoning_effort": "high",
    "stream": False
}

url = "https://api.deepseek.com/v1/chat/completions"
response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print(response.json()["choices"][0]["message"]["content"])
else:
    print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")

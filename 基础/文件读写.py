import json

data = {
    "model": "deepseek-v4-pro",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "thinking": {"type": "enabled"},
    "reasoning_effort": "high",
    "stream": False
}

with open("test.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(data, ensure_ascii=False))


with open("test.txt", "r", encoding="utf-8") as f:
    json_data = json.loads(f.read())
    print(json_data)
    print(type(json_data))
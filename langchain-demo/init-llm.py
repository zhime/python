import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
dp_llm = init_chat_model(
    model='deepseek-v4-flash',
    model_provider='deepseek',
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url=os.environ.get('DEEPSEEK_BASE_URL'),
)

# resp = dp_llm.invoke("你是哪个模型")
# print(resp.content)

resp = dp_llm.stream("你是哪个模型")
for item in resp:
    print(item.content, end='')

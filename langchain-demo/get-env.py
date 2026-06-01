import os

from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.environ.get('DEEPSEEK_BASE_URL')

print(DEEPSEEK_API_KEY)
print(DEEPSEEK_BASE_URL)


"""
Test: Bedrock LLM Proxy — test all available models
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BEDROCK_BASE_URL")
API_KEY = os.getenv("BEDROCK_API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

models_to_test = ["sonnet", "nova-micro", "nova-lite", "nova-pro"]

for model in models_to_test:
    print(f"\n--- Testing: {model} ---")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            max_tokens=50,
        )
        print(f"  ✓ Response: {resp.choices[0].message.content}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

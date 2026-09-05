import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env (expects GROQ_API_KEY)
load_dotenv()

# Active text models available on your Groq account:
# - "qwen/qwen3.6-27b"
# - "qwen/qwen3.8-27b"
# - "openai/gpt-oss-120b"
# - "openai/gpt-oss-20b"

# Initialize ChatGroq
llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.7,
)

# Invoke the model
response = llm.invoke("What is the capital of France?")

# Print response
print(response.content)
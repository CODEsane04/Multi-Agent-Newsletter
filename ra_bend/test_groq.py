import os
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq

load_dotenv()

# Step 1: List all active models available for your Groq API key
print("Fetching available models from Groq...")
groq_client = Groq()
models = [m.id for m in groq_client.models.list().data]
print(f"Available Groq models: {models}\n")

# Use the first active model or a standard model like llama3-8b-8192 / mixtral-8x7b-32768
target_model = models[0] if models else "llama3-8b-8192"
print(f"Testing model: {target_model}")

# Step 2: Test via LangChain ChatGroq
llm = ChatGroq(
    model=target_model,
    temperature=0.7,
)

response = llm.invoke("What is the capital of France?")
print(f"\nResponse:\n{response.content}")

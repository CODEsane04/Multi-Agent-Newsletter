"""
Test: Single item classification using Gemma via LangChain
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0.1)
parser = StrOutputParser()
chain = model | parser

prompt = """You are a news classifier. Classify this news item into exactly one class:

Classes:
1. Release of a new frontier model or an improved model
2. New Academic research or paper
3. New AI product or Application launch
4. Hardware or Infrastructure
5. Business policy and Geopolitics
6. General or miscellaneous

News item: OpenAI releases GPT-5 with 256k context window and new reasoning capabilities — Benchmarks show 95% on MMLU

Respond with ONLY the class number. Nothing else."""

print("Sending request to Gemma...")
result = chain.invoke(prompt)
print(f"Response: {result}")

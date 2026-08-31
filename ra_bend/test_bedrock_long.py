"""
Test: Bedrock Sonnet with a long prompt to check if connection holds.
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BEDROCK_BASE_URL")
API_KEY = os.getenv("BEDROCK_API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0)

long_prompt = """You are a senior AI journalist. Write a detailed 200-word newsletter section about the following news:

Meta CEO Mark Zuckerberg told staff during an internal town hall that AI agent development has not progressed as quickly as executives had hoped. The company has spent billions on AI infrastructure, laid off thousands, and reassigned 7,000 employees to AI groups including a unit called "Agent Transformation." Despite this massive investment, functional AI agents that can autonomously complete business tasks remain elusive. Zuckerberg expects improvements within 3-6 months but acknowledged the gap between investment and results. Engineers in the new AI units report high stress levels and unclear objectives.

Write the section with:
1. A headline (max 10 words)
2. A TL;DR (one sentence)
3. What happened (2-3 sentences)
4. Why it matters (2-3 sentences)
5. Key details (3-5 bullet points)

Be concise and technical. No fluff."""

print("Sending long prompt to Bedrock Sonnet...")
try:
    resp = client.chat.completions.create(
        model="sonnet",
        messages=[{"role": "user", "content": long_prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    print(f"✓ Success!\n")
    print(resp.choices[0].message.content)
except Exception as e:
    print(f"✗ Failed: {e}")

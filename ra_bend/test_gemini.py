import os
from langchain_xai import ChatXAI
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

load_dotenv()

# 1. Set your xAI API key via .env (XAI_API_KEY)
api_key = os.getenv("XAI_API_KEY")

# 2. Initialize the Grok model
# Common models include "grok-beta", "grok-2", or "grok-2-latest"
llm = ChatXAI(
model="grok-beta",
temperature=0.7,
# max_tokens=1024, # Optional: control response length
)

# 3. Create a message and invoke the model
messages = [
HumanMessage(content="Write a short, funny haiku about debugging code.")
]

# 4. Get the response
response = llm.invoke(messages)
print(response.content)

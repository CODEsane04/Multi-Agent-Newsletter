from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    temperature=0.1
)

response = llm.invoke("what is the capirtal of france")
print(response.content)
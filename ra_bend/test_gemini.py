import warnings
import logging

# Suppress deprecation and Google SDK warnings
warnings.filterwarnings("ignore")
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    temperature=0.1
)

parser = StrOutputParser()

chain = llm | parser
print("\n")
response = chain.invoke("what is dark matter, exlain in 1 sentence? ")
print(response)
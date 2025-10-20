
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

llm=ChatOpenAI(
    model=os.getenv("MODEL_NAME","llama3.1-8b"),
    openai_api_key=os.environ["CEREBRAS_API_KEY"],
    openai_api_base=os.getenv("CEREBRAS_API_BASE","https://api.cerebras.ai/v1"),
    temperature=0.2,
)

prompt=ChatPromptTemplate.from_messages([
    ("system","Du bist ein höflicher, präziser Hotel-Concierge. Antworte kurz."),
    ("human","{frage}")
])

chain=prompt|llm|StrOutputParser()

def ask(q:str)->str:
    return chain.invoke({"frage":q})

if __name__=="__main__":
    for f in [
        "Wann ist der Check-in?",
        "Gibt es Parkplätze?",
        "Wie komme ich mit dem Bus zum Flughafen?",
        "Wie lautet das WLAN-Passwort?"
    ]:
        print("Gast:",f)
        print("Concierge:",ask(f))
        print("-"*50)

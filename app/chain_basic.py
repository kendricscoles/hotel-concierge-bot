from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.prompts import SYSTEM_PROMPT
from app.config import MODEL_NAME

try:
    from langchain_community.chat_models import ChatCerebras
except ImportError:
    from langchain_community.llms import Cerebras as ChatCerebras

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{user_input}")
])

llm = ChatCerebras(model=MODEL_NAME, temperature=0.2)

chain = prompt | llm | StrOutputParser()

def ask_once(text: str) -> str:
    return chain.invoke({"user_input": text})

def demo():
    examples = [
        "Ab wann ist Check-in?",
        "Wie lautet das WLAN-Passwort?",
        "Gibt es Parkplätze und was kosten sie?",
        "Kann ich einen Late-Checkout bis 13:30 machen?"
    ]
    for q in examples:
        ans = ask_once(q)
        print(f"Gast: {q}\nConcierge: {ans}\n" + "-"*60)

if __name__ == "__main__":
    demo()

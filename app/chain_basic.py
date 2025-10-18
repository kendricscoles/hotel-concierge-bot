from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.prompts import SYSTEM_PROMPT
from app.config import (
    MODEL_NAME,
    CEREBRAS_API_KEY,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
)

lf = None
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    try:
        from langfuse import Langfuse
        lf = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    except Exception as e:
        print("Langfuse initialization failed:", e)
        lf = None

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{user_input}")
])

llm = ChatOpenAI(
    model=gpt-oss-120b,
    api_key=CEREBRAS_API_KEY,
    base_url="https://api.cerebras.ai/v1",
    temperature=0.2,
)

chain = prompt | llm | StrOutputParser()

def ask_once(text: str) -> str:
    if lf:
        with lf.trace(name="concierge-ask-once", metadata={"provider": "cerebras", "model": MODEL_NAME}) as trace:
            trace.update(input=text)
            with trace.span(name="llm-chain"):
                out = chain.invoke({"user_input": text})
            trace.update(output=out)
            return out
    return chain.invoke({"user_input": text})

def demo():
    print("=== Hotel Concierge Mini-Demo (Cerebras) ===\n")
    examples = [
        "Ab wann ist Check-in?",
        "Wie lautet das WLAN-Passwort?",
        "Gibt es Parkplätze und was kosten sie pro Nacht?",
        "Kann ich einen Late-Checkout bis 13:30 machen?",
    ]
    for q in examples:
        ans = ask_once(q)
        print(f"Gast: {q}\nConcierge: {ans}\n" + "-"*60)

if __name__ == "__main__":
    demo()

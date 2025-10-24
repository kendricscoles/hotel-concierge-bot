import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.schema.runnable import RunnablePassthrough
from app.tools_concierge import web_search, wiki_search, current_weather, fx_convert

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b")
BASE_URL = os.getenv("OPENAI_BASE_URL", os.getenv("CEREBRAS_API_BASE", "https://api.cerebras.ai/v1"))
API_KEY = os.environ["CEREBRAS_API_KEY"]

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.2,
)

tools = [web_search, wiki_search, current_weather, fx_convert]
llm_with_tools = llm.bind_tools(tools=tools, tool_choice="auto", strict=False)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a hotel concierge assistant for Basel. Use tools when needed. Prefer concise, practical answers. If a tool was used, state which one in parentheses at the end."),
    ("human", "{question}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

chain = (
    {
        "question": RunnablePassthrough(),
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x.get("intermediate_steps", [])),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent = AgentExecutor(agent=chain, tools=tools, verbose=False)

def answer_with_tools(question: str) -> str:
    result = agent.invoke({"question": question})
    return result.get("output", "").strip()
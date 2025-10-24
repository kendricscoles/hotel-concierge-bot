import os
import json
import ast
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from app.tools_concierge import web_search, wiki_search, current_weather, fx_convert

try:
    ChatOpenAI.model_rebuild()
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_API_BASE = os.getenv("CEREBRAS_API_BASE", "https://api.cerebras.ai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b")
if not CEREBRAS_API_KEY:
    raise RuntimeError("Missing CEREBRAS_API_KEY in .env")

os.environ["OPENAI_API_KEY"] = CEREBRAS_API_KEY
os.environ.pop("OPENAI_BASE_URL", None)

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=CEREBRAS_API_KEY,
    base_url=CEREBRAS_API_BASE,
    temperature=0.2,
    max_retries=0,
)

tools = [web_search, wiki_search, current_weather, fx_convert]
llm_with_tools = llm.bind_tools(tools=tools, tool_choice="auto", strict=False)
tool_map = {t.name: t for t in tools}

GERMAN_SYSTEM = (
    "Du bist der Concierge eines 5-Sterne-Luxushotels in Basel. "
    "Du darfst nur auf Basis von echten, geprüften Informationen antworten. "
    "Wenn du etwas nicht sicher weißt oder kein Ergebnis aus einem Tool erhältst, "
    "sage klar, dass keine Daten verfügbar sind – erfinde niemals etwas. "
    "Antworte auf Deutsch, in kurzen, eleganten und präzisen Sätzen. "
    "Verwende, wann immer möglich, Informationen aus deinen Tools "
    "(Wetter, Währungsrechner, Wikipedia, Websuche). "
    "Keine Erwähnung von Tools, Quellen oder internen Abläufen."
)

def _invoke_llm_with_backoff(messages, attempts: int = 6):
    for i in range(attempts):
        try:
            return llm_with_tools.invoke(messages)
        except Exception as e:
            emsg = str(e)
            klass = e.__class__.__name__
            retryable = (
                "too_many_requests" in emsg.lower()
                or "queue_exceeded" in emsg.lower()
                or "ratelimiterror" in emsg.lower()
                or klass.lower() == "ratelimiterror"
            )
            if not retryable:
                raise
            base = min(8.0, 0.75 * (2 ** i))
            jitter = 0.7 + 0.6 * random.random()
            time.sleep(base * jitter)
    raise RuntimeError("Der Dienst ist vorübergehend ausgelastet. Bitte gleich nochmals versuchen.")

def _maybe_parse_json(s):
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None

def _extract_tool_calls(ai_msg):
    calls = []
    tc = getattr(ai_msg, "tool_calls", None)
    if isinstance(tc, list) and tc:
        for c in tc:
            calls.append({"id": c.get("id", ""), "name": c.get("name"), "args": c.get("args")})
        if calls:
            return calls
    ak = getattr(ai_msg, "additional_kwargs", {}) or {}
    tc2 = ak.get("tool_calls")
    if isinstance(tc2, list) and tc2:
        for c in tc2:
            fn = c.get("function", {}) or {}
            calls.append({"id": c.get("id", ""), "name": fn.get("name"), "args": fn.get("arguments")})
        if calls:
            return calls
    content = getattr(ai_msg, "content", None)
    if isinstance(content, str) and content:
        obj = _maybe_parse_json(content)
        if obj is not None:
            seq = obj if isinstance(obj, list) else [obj]
            for o in seq:
                if not isinstance(o, dict):
                    continue
                name = o.get("name") or (o.get("function") or {}).get("name")
                args = o.get("arguments") or o.get("args") or o.get("parameters") or (
                    (o.get("function") or {}).get("arguments")
                )
                if name and args is not None:
                    calls.append({"id": o.get("id", ""), "name": name, "args": args})
            if calls:
                return calls
        for line in content.splitlines():
            line = line.strip()
            if not (line.startswith("{") and line.endswith("}")):
                continue
            o = _maybe_parse_json(line)
            if isinstance(o, dict):
                name = o.get("name") or (o.get("function") or {}).get("name")
                args = o.get("arguments") or o.get("args") or o.get("parameters") or (
                    (o.get("function") or {}).get("arguments")
                )
                if name and args is not None:
                    calls.append({"id": o.get("id", ""), "name": name, "args": args})
        if calls:
            return calls
    return calls

def _coerce_args(args):
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        obj = _maybe_parse_json(args)
        if isinstance(obj, dict):
            return obj
        return {"__raw__": args}
    return {"__raw__": str(args)}

def _polish(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    forbidden = [
        "tool",
        "quelle",
        "wikipedia",
        "duckduckgo",
        "open-meteo",
        "exchangerate.host",
        "frankfurter",
        "installieren",
        "nicht verfügbar",
    ]
    lower = t.lower()
    if any(word in lower for word in forbidden):
        lines = [ln for ln in t.splitlines() if not any(w in ln.lower() for w in forbidden)]
        t = " ".join(lines).strip()
    t = t.replace("  ", " ").strip()
    if t and not t.endswith("."):
        t += "."
    return t

def answer_with_tools(question: str) -> str:
    messages = [SystemMessage(content=GERMAN_SYSTEM), HumanMessage(content=question)]
    for _ in range(8):
        ai = _invoke_llm_with_backoff(messages)
        calls = _extract_tool_calls(ai)
        if calls:
            messages.append(ai)
            for call in calls:
                name = call.get("name")
                args = _coerce_args(call.get("args"))
                tool = tool_map.get(name)
                if tool is None:
                    messages.append(ToolMessage(content="", tool_call_id=call.get("id", "")))
                    continue
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    result = f"Fehler: {e}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call.get("id", "")))
            continue
        text = (getattr(ai, "content", "") or "").strip()
        if text.lower().startswith(("ich kann", "ich werde", "lass mich", "ich helfe", "gerne")):
            continue
        if text:
            return _polish(text)
    return "Entschuldigung, das hat nicht geklappt."
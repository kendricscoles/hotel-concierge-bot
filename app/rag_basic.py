import os, pathlib, warnings, time
from typing import List, Tuple
from pathlib import Path
from dotenv import load_dotenv
from langsmith import traceable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, BSHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")

warnings.filterwarnings("ignore")
os.environ.setdefault("USER_AGENT", os.getenv("USER_AGENT", "hotel-concierge-bot/1.0"))

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_API_BASE = os.getenv("CEREBRAS_API_BASE", "https://api.cerebras.ai/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", CEREBRAS_API_BASE)

DATA_DIR = str(ROOT / "data")
INDEX_DIR = str(ROOT / "index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K_DEFAULT = int(os.getenv("TOP_K", "4"))

_client = None
_embeddings = None
_vs_persisted = None

def _client_openai() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    api_key = CEREBRAS_API_KEY or OPENAI_API_KEY
    base_url = OPENAI_BASE_URL or CEREBRAS_API_BASE
    if not api_key:
        raise RuntimeError("No API key set.")
    _client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    return _client

def _load_dir_documents(dir_path: str):
    docs = []
    p = pathlib.Path(dir_path)
    if not p.exists():
        return docs
    for f in sorted(p.glob("*")):
        sfx = f.suffix.lower()
        try:
            if sfx == ".pdf":
                try:
                    docs += PyMuPDFLoader(str(f)).load()
                except Exception:
                    docs += PyPDFLoader(str(f)).load()
            elif sfx in {".html", ".htm"}:
                docs += BSHTMLLoader(str(f)).load()
            elif sfx in {".txt", ".md", ".markdown"}:
                docs += TextLoader(str(f), encoding="utf-8").load()
        except Exception:
            pass
    return docs

def _emb():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings

def _persisted_vs():
    global _vs_persisted
    if _vs_persisted is not None:
        return _vs_persisted
    pathlib.Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    try:
        _vs_persisted = FAISS.load_local(INDEX_DIR, _emb(), allow_dangerous_deserialization=True)
        return _vs_persisted
    except Exception:
        pass
    base_docs = _load_dir_documents(DATA_DIR)
    if not base_docs:
        _vs_persisted = FAISS.from_texts([""], _emb())
        _vs_persisted.save_local(INDEX_DIR)
        return _vs_persisted
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(base_docs)
    _vs_persisted = FAISS.from_documents(chunks, _emb())
    _vs_persisted.save_local(INDEX_DIR)
    return _vs_persisted

SYSTEM_PROMPT = (
    "Du bist ein präziser, freundlicher Hotel-Concierge in Basel. Antworte knapp, sachlich und hilfreich. "
    "Nutze den gegebenen Kontext für Fakten. Wenn dir im Kontext etwas fehlt, antworte trotzdem so gut wie möglich "
    "und kennzeichne allgemeine Hinweise mit allgemein."
)

def _format_context(chunks: List[Tuple[str, dict]]) -> str:
    lines = []
    for (text, meta) in chunks:
        snippet = (text or "").strip().replace("\n", " ")
        if len(snippet) > 900:
            snippet = snippet[:900] + " ..."
        lines.append(snippet)
    return "\n\n".join(lines)

@traceable(name="chat_call")
def _chat(prompt: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
    client = _client_openai()
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (r.choices[0].message.content or "").strip()

@traceable(name="retrieve")
def retrieve(query: str, k: int = TOP_K_DEFAULT):
    q = (query or "").strip()
    if not q:
        return []
    base_vs = _persisted_vs()
    try:
        return [(d.page_content, d.metadata) for d in base_vs.similarity_search(q, k=k)]
    except Exception:
        time.sleep(0.5)
        base_vs = _persisted_vs()
        return [(d.page_content, d.metadata) for d in base_vs.similarity_search(q, k=k)]

@traceable(name="answer_with_llm")
def answer_with_llm(query: str, k: int = TOP_K_DEFAULT) -> str:
    q = (query or "").strip()
    if not q:
        return "Bitte stelle eine Frage."
    chunks = retrieve(q, k=k)
    if chunks:
        ctx = _format_context(chunks)
        prompt = f"Frage:\n{q}\n\nKontext:\n{ctx}\n\nHinweise:\n- Antworte auf Deutsch.\n- Antworte nur basierend auf dem Kontext."
        return _chat(prompt)
    prompt = f"Frage (ohne lokale Quellen):\n{q}\n\nAntworte kurz auf Deutsch. Markiere allgemeine Hinweise mit allgemein."
    return _chat(prompt)

def debug_list_sources():
    base_vs = _persisted_vs()
    try:
        index_info = base_vs.index.ntotal
    except Exception:
        index_info = None
    docs = _load_dir_documents(DATA_DIR)
    names = []
    for d in docs:
        m = getattr(d, "metadata", {}) or {}
        names.append(m.get("source") or m.get("file_path") or "unknown")
    return {"faiss_ntotal": index_info, "loaded_docs": sorted(set(pathlib.Path(n).name for n in names))}
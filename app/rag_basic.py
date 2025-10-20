import os, pathlib, warnings
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

load_dotenv()
warnings.filterwarnings("ignore")
os.environ.setdefault("USER_AGENT", os.getenv("USER_AGENT", "hotel-concierge-bot/1.0"))

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_API_BASE = os.getenv("CEREBRAS_API_BASE", "https://api.cerebras.ai/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", CEREBRAS_API_BASE)
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = str(ROOT / "data")
INDEX_DIR = str(ROOT / "index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K_DEFAULT = int(os.getenv("TOP_K", "4"))

_client: Optional[OpenAI] = None
def _client_openai() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    api_key = CEREBRAS_API_KEY or OPENAI_API_KEY
    base_url = OPENAI_BASE_URL or CEREBRAS_API_BASE
    if not api_key:
        raise RuntimeError("No API key set.")
    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client

def _load_dir_documents(dir_path: str) -> List:
    docs = []
    p = pathlib.Path(dir_path)
    if not p.exists():
        return docs
    for f in sorted(p.glob("*")):
        sfx = f.suffix.lower()
        try:
            if sfx == ".pdf":
                docs += PyPDFLoader(str(f)).load()
            elif sfx in {".html", ".htm"}:
                docs += BSHTMLLoader(str(f)).load()
            elif sfx in {".txt", ".md", ".markdown"}:
                docs += TextLoader(str(f), encoding="utf-8").load()
        except Exception:
            pass
    return docs

_embeddings = None
_vs_persisted = None

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
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(base_docs)
    _vs_persisted = FAISS.from_documents(chunks, _emb())
    _vs_persisted.save_local(INDEX_DIR)
    return _vs_persisted

def _temp_vs_from_uploads(paths: List[str]):
    docs = []
    for p in (paths or []):
        f = pathlib.Path(p)
        if not f.exists():
            continue
        try:
            if f.suffix.lower() == ".pdf":
                docs += PyPDFLoader(str(f)).load()
            elif f.suffix.lower() in {".html", ".htm"}:
                docs += BSHTMLLoader(str(f)).load()
            elif f.suffix.lower() in {".txt", ".md", ".markdown"}:
                docs += TextLoader(str(f), encoding="utf-8").load()
        except Exception:
            pass
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, _emb())

def retrieve(query: str, k: int = TOP_K_DEFAULT, uploaded_paths: Optional[List[str]] = None) -> List[Tuple[str, dict]]:
    q = (query or "").strip()
    if not q:
        return []
    base_vs = _persisted_vs()
    base_hits = base_vs.similarity_search(q, k=k)
    if uploaded_paths:
        tmp_vs = _temp_vs_from_uploads(uploaded_paths)
        if tmp_vs is not None:
            tmp_hits = tmp_vs.similarity_search(q, k=k)
            merged = []
            for a, b in zip(base_hits, tmp_hits):
                merged.extend([a, b])
            longer = base_hits if len(base_hits) > len(tmp_hits) else tmp_hits
            merged.extend(longer[len(merged)//2:])
            hits = merged[:k]
        else:
            hits = base_hits
    else:
        hits = base_hits
    return [(d.page_content, d.metadata) for d in hits]

SYSTEM_PROMPT = (
    "Du bist ein präziser, freundlicher Hotel-Concierge in Basel. Antworte knapp, sachlich und hilfreich. "
    "Nutze nur den gegebenen Kontext für Fakten. Wenn dir im Kontext etwas fehlt, sag das explizit und antworte "
    "ggf. mit allgemeinen Hinweisen, markiert als allgemein. Zitiere Quellen als [S1], [S2], ..."
)

def _format_context(chunks: List[Tuple[str, dict]]) -> str:
    lines = []
    for i, (text, meta) in enumerate(chunks, start=1):
        src = meta.get("source") or meta.get("file_path") or meta.get("source_id") or "unknown"
        page = meta.get("page", None)
        label = f"{src}" + (f":p{page}" if page is not None else "")
        snippet = (text or "").strip().replace("\n", " ")
        if len(snippet) > 900:
            snippet = snippet[:900] + " ..."
        lines.append(f"[S{i}] {label}\n{snippet}")
    return "\n\n".join(lines)

def _chat(prompt: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
    client = _client_openai()
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (r.choices[0].message.content or "").strip()

def answer_with_llm(query: str, k: int = TOP_K_DEFAULT, uploaded_paths: Optional[List[str]] = None) -> Tuple[str, str]:
    q = (query or "").strip()
    if not q:
        return "Bitte stelle eine Frage.", ""
    chunks = retrieve(q, k=k, uploaded_paths=uploaded_paths)
    if chunks:
        ctx = _format_context(chunks)
        prompt = f"Frage:\n{q}\n\nKontext:\n{ctx}\n\nHinweise:\n- Antworte auf Deutsch.\n- Zitiere [S1], [S2], ..."
        ans = _chat(prompt)
        srcs = []
        for i, (_, meta) in enumerate(chunks, start=1):
            src = meta.get("source") or meta.get("file_path") or "unknown"
            page = meta.get("page", "")
            tag = f"[S{i}]"
            srcs.append(f"{tag} {pathlib.Path(src).name}{(':'+str(page)) if page!='' else ''}")
        return ans, "; ".join(srcs)
    prompt = f"Frage (ohne lokale Quellen):\n{q}\n\nAntworte kurz auf Deutsch. Markiere allgemeine Hinweise mit allgemein."
    return _chat(prompt), "(keine lokalen Quellen)"

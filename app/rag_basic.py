import os, pathlib, warnings, time, re, unicodedata
from typing import List, Tuple
from pathlib import Path
from dotenv import load_dotenv
from langsmith import traceable

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, BSHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from app.agent_tools_fallback import answer_with_tools

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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K_DEFAULT = int(os.getenv("TOP_K", "4"))
PREFERRED_HOTEL = os.getenv("HOTEL_NAME", "").strip().lower()

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


def _latest_mtime_in_dir(p: pathlib.Path) -> float:
    if not p.exists():
        return 0.0
    latest = 0.0
    for f in p.rglob("*"):
        try:
            latest = max(latest, f.stat().st_mtime)
        except Exception:
            pass
    return latest


def _build_vs_from_data():
    base_docs = _load_dir_documents(DATA_DIR)
    if not base_docs:
        return None
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(base_docs)
    return FAISS.from_documents(chunks, _emb())


def _persisted_vs():
    global _vs_persisted
    if _vs_persisted is not None:
        return _vs_persisted
    index_path = pathlib.Path(INDEX_DIR)
    data_path = pathlib.Path(DATA_DIR)
    index_path.mkdir(parents=True, exist_ok=True)
    data_mtime = _latest_mtime_in_dir(data_path)
    index_mtime = _latest_mtime_in_dir(index_path)
    loaded = None
    if any(index_path.iterdir()):
        try:
            loaded = FAISS.load_local(INDEX_DIR, _emb(), allow_dangerous_deserialization=True)
        except Exception:
            loaded = None
    if loaded is not None:
        try:
            ntotal = loaded.index.ntotal
        except Exception:
            ntotal = 0
        if ntotal > 0 and index_mtime >= data_mtime:
            _vs_persisted = loaded
            return _vs_persisted
    built = _build_vs_from_data()
    if built is None:
        _vs_persisted = None
        return _vs_persisted
    _vs_persisted = built
    try:
        _vs_persisted.save_local(INDEX_DIR)
    except Exception:
        pass
    return _vs_persisted


SYSTEM_PROMPT = (
    "Du bist ein präziser, freundlicher Hotel-Concierge in Basel. "
    "Antworte bevorzugt mit Informationen aus den lokalen Hotelunterlagen. "
    "Wenn keine lokalen Daten passen, darfst du allgemeines Wissen verwenden; sei dabei knapp, hilfreich und korrekt."
)

SYSTEM_PROMPT_RAG_ONLY = (
    "Du bist ein präziser, freundlicher Hotel-Concierge in Basel. "
    "Antworte ausschließlich mit Informationen aus dem gegebenen Kontext."
)

SYSTEM_PROMPT_OPEN = (
    "Du bist ein präziser, freundlicher Hotel-Concierge in Basel. "
    "Wenn keine lokalen Unterlagen vorliegen, nutze dein allgemeines Wissen, antworte knapp und korrekt auf Deutsch."
)


def _format_context(chunks: List[Tuple[str, dict]]) -> str:
    lines = []
    for (text, meta) in chunks:
        snippet = (text or "").strip().replace("\n", " ")
        if len(snippet) > 900:
            snippet = snippet[:900] + " ..."
        lines.append(snippet)
    return "\n\n".join(lines)


def _src_name(meta: dict) -> str:
    src = (meta or {}).get("source") or (meta or {}).get("file_path") or ""
    return pathlib.Path(src).name.lower()


def _is_hotel_doc(name: str) -> bool:
    n = name.lower()
    if any(w in n for w in ["hotel", "gaesteinformation", "gästeinformation", "factsheet", "guest", "guests", "info"]):
        return True
    return False


def _hotel_docs_in_data() -> List[str]:
    p = pathlib.Path(DATA_DIR)
    if not p.exists():
        return []
    names = [f.name.lower() for f in p.glob("*") if
             f.suffix.lower() in [".pdf", ".html", ".htm", ".txt", ".md", ".markdown"]]
    return [n for n in names if _is_hotel_doc(n)]


def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _expand_queries(q: str) -> List[str]:
    n = _norm(q)
    qs = {q.strip()}
    if "fruhstuck" in n or "fruehstueck" in n or "fruhstueck" in n or "fruh" in n:
        qs.update(
            {"Frühstück", "fruehstueck", "fruhstuck", "Breakfast", "breakfast", "petit dejeuner", "petit-déjeuner"})
    if "check in" in n or "checkin" in n or "eincheck" in n:
        qs.update({"Check-in", "check in", "checkin", "arrival", "anreise"})
    if "check out" in n or "checkout" in n or "auscheck" in n:
        qs.update({"Check-out", "check out", "checkout", "departure", "abreise"})
    if "spa" in n or "wellness" in n:
        qs.update({"spa", "wellness", "massage", "sauna"})
    return sorted({s for s in qs if s})


def _prioritize_list(docs, only_hotel: bool):
    hotel_names = _hotel_docs_in_data()
    single_hotel_mode = len(hotel_names) == 1
    only_name = hotel_names[0] if single_hotel_mode else ""
    preferred = PREFERRED_HOTEL

    def score(d):
        name = _src_name(d.metadata)
        if only_hotel and not _is_hotel_doc(name):
            return 10_000
        s = 0
        if _is_hotel_doc(name):
            s += 40
        if single_hotel_mode and only_name and only_name in name:
            s += 40
        if preferred and preferred in name:
            s += 35
        if "basel" in name:
            s += 5
        if any(b in name for b in ("tram", "strassenbahn", "bahn", "tnw", "tarifverbund", "baustelle", "bvb")):
            s -= 40
        return -s

    return sorted(docs, key=score)


def _keyword_fallback(query: str, k: int, only_hotel: bool):
    tokens = [t for t in re.split(r"\W+", _norm(query)) if t]
    if not tokens:
        return []
    all_docs = _load_dir_documents(DATA_DIR)
    if not all_docs:
        return []
    hits = []
    for d in all_docs:
        src = _src_name(getattr(d, "metadata", {}) or {})
        if only_hotel and not _is_hotel_doc(src):
            continue
        text = _norm(getattr(d, "page_content", ""))
        if any(t in text for t in tokens):
            hits.append(d)
    hits = _prioritize_list(hits, only_hotel)
    return [(d.page_content, d.metadata) for d in hits[:k]]


def _direct_pattern_lookup(query: str, k: int, only_hotel: bool):
    qn = _norm(query)
    synonyms = set()
    if any(t in qn for t in ["fruhstuck", "fruehstueck", "fruhstueck", "fruh", "fruhstück"]):
        synonyms.update(["frühstück", "fruehstueck", "fruhstuck", "breakfast", "petit dejeuner", "petit-déjeuner"])
    if any(t in qn for t in ["check in", "checkin", "eincheck", "anreise"]):
        synonyms.update(["check-in", "arrival", "einchecken", "check in"])
    if any(t in qn for t in ["check out", "checkout", "auscheck", "abreise"]):
        synonyms.update(["check-out", "departure", "checkout", "auschecken"])
    if any(t in qn for t in ["spa", "wellness", "massage", "sauna"]):
        synonyms.update(["spa", "wellness", "massage", "sauna"])
    if not synonyms:
        return []
    docs = _load_dir_documents(DATA_DIR)
    if not docs:
        return []
    hits = []
    pattern = re.compile(r"(?i)(\b(?:%s)\b).{0,160}?(\d{1,2}[:.]\d{2})" % "|".join(map(re.escape, synonyms)))
    for d in docs:
        src = _src_name(getattr(d, "metadata", {}) or {})
        if only_hotel and not _is_hotel_doc(src):
            continue
        text = getattr(d, "page_content", "") or ""
        for match in pattern.finditer(text):
            snippet = text[max(0, match.start() - 80): match.end() + 80]
            hits.append((snippet.strip(), d.metadata))
    if not hits:
        time_pattern = re.compile(r"(?i)\b\d{1,2}[:.]\d{2}\b")
        for d in docs:
            src = _src_name(getattr(d, "metadata", {}) or {})
            if only_hotel and not _is_hotel_doc(src):
                continue
            text = getattr(d, "page_content", "") or ""
            if any(s in text.lower() for s in synonyms) and time_pattern.search(text):
                snippet = text[:600]
                hits.append((snippet.strip(), d.metadata))
    return hits[:k]


def _deep_scan(query: str, k: int):
    hotel_first = len(_hotel_docs_in_data()) > 0
    direct = _direct_pattern_lookup(query, k, only_hotel=hotel_first)
    if direct:
        return direct
    kw = _keyword_fallback(query, k, only_hotel=hotel_first)
    if kw:
        return kw
    direct2 = _direct_pattern_lookup(query, k, only_hotel=False)
    if direct2:
        return direct2
    kw2 = _keyword_fallback(query, k, only_hotel=False)
    return kw2


@traceable(name="chat_call")
def _chat(prompt: str, temperature: float = 0.2, max_tokens: int = 700, system_prompt: str = SYSTEM_PROMPT) -> str:
    client = _client_openai()
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
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
    hotel_docs_exist = len(_hotel_docs_in_data()) > 0
    if base_vs is None:
        return _deep_scan(q, k)
    queries = _expand_queries(q) or [q]
    pool = []
    for subq in queries:
        try:
            pool += base_vs.similarity_search(subq, k=max(k * 3, k + 4))
        except Exception:
            time.sleep(0.05)
    if not pool:
        return _deep_scan(q, k)
    hotel_pool = [d for d in pool if _is_hotel_doc(_src_name(d.metadata))]
    other_pool = [d for d in pool if not _is_hotel_doc(_src_name(d.metadata))]
    if hotel_docs_exist and hotel_pool:
        hotel_pool = _prioritize_list(hotel_pool, only_hotel=True)
        return [(d.page_content, d.metadata) for d in hotel_pool[:k]]
    combined = _prioritize_list(pool, only_hotel=False)
    if combined:
        return [(d.page_content, d.metadata) for d in combined[:k]]
    return _deep_scan(q, k)


@traceable(name="answer_with_llm")
def answer_with_llm(query: str, k: int = TOP_K_DEFAULT) -> str:
    q = (query or "").strip()
    if not q:
        return "Bitte stelle eine Frage."
    chunks = retrieve(q, k=k)
    if chunks:
        ctx = _format_context(chunks)
        prompt = f"Frage:\n{q}\n\nKontext (lokale Unterlagen):\n{ctx}\n\nAntworte nur basierend auf dem Kontext."
        return _chat(prompt, system_prompt=SYSTEM_PROMPT_RAG_ONLY)
    deep = _deep_scan(q, k=max(k, 6))
    if deep:
        ctx = _format_context(deep)
        prompt = f"Frage:\n{q}\n\nKontext (Deep-Scan lokale Unterlagen):\n{ctx}\n\nAntworte nur basierend auf dem Kontext."
        return _chat(prompt, system_prompt=SYSTEM_PROMPT_RAG_ONLY)
    return answer_with_tools(q)


def debug_list_sources():
    base_vs = _persisted_vs()
    index_info = None
    if base_vs is not None:
        try:
            index_info = base_vs.index.ntotal
        except Exception:
            index_info = None
    docs = _load_dir_documents(DATA_DIR)
    names = []
    for d in docs:
        m = getattr(d, "metadata", {}) or {}
        names.append(m.get("source") or m.get("file_path") or "unknown")
    return {
        "faiss_ntotal": index_info,
        "data_dir": DATA_DIR,
        "index_dir": INDEX_DIR,
        "loaded_docs": sorted(set(pathlib.Path(n).name for n in names)),
    }
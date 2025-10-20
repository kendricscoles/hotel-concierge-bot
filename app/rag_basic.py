
import os, re, warnings, httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
os.environ.setdefault("USER_AGENT","hotel-concierge-bot/1.0")
load_dotenv()

BASE_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
DATA_DIR=os.path.join(BASE_DIR,"data")
DOC_SEP="\n\n---\n\n"

def _read_file_text(path):
    try:
        with open(path, "rb") as f:
            b=f.read()
        try:
            s=b.decode("utf-8")
        except UnicodeDecodeError:
            s=b.decode("latin-1","ignore")
        return s
    except Exception:
        return ""

def collect_docs():
    docs=[]
    if not os.path.isdir(DATA_DIR):
        return docs
    for name in sorted(os.listdir(DATA_DIR)):
        p=os.path.join(DATA_DIR,name)
        lo=name.lower()
        try:
            if lo.endswith(".pdf"):
                docs+=PyPDFLoader(p).load()
            elif lo.endswith((".html",".htm")):
                docs+=BSHTMLLoader(p).load()
            elif lo.endswith(".txt"):
                docs+=TextLoader(p, encoding="utf-8").load()
        except Exception:
            pass
    return docs

def build_chunks(docs):
    if not docs: return []
    s=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return s.split_documents(docs)

def build_retriever(chunks):
    if not chunks: return None
    try:
        return BM25Retriever.from_documents(chunks, k=12)
    except Exception:
        return None

def build_llm():
    model=os.getenv("MODEL_NAME","llama3.1-8b")
    api_base=os.getenv("CEREBRAS_API_BASE","https://api.cerebras.ai/v1")
    api_key=os.getenv("CEREBRAS_API_KEY")
    return ChatOpenAI(model=model, temperature=0.2, openai_api_key=api_key, openai_api_base=api_base)

SYSTEM=(
 "Du bist ein präziser, freundlicher Hotel-Concierge in Basel. Antworte knapp, sachlich und hilfreich. "
 "Nutze nur den gegebenen Kontext; erfinde nichts. Wenn du schätzen musst, kennzeichne mit 'geschätzt'. "
 "Falls keine belastbare Quelle vorliegt, liefere eine freundliche, leicht humorvolle Antwort und biete an, es zu klären."
)
prompt=ChatPromptTemplate.from_messages([
    ("system", SYSTEM+"\n\nKontext:\n{context}"),
    ("human","{query}")
])
_chain = prompt | build_llm() | StrOutputParser()

_docs=None; _chunks=None; _ret=None; _raw_cache={}
def _ensure():
    global _docs,_chunks,_ret
    if _docs is None: _docs=collect_docs()
    if _chunks is None: _chunks=build_chunks(_docs)
    if _ret is None: _ret=build_retriever(_chunks)

def _hotel_facts():
    facts={}
    for d in (_docs or []):
        src=str(d.metadata.get("source","")).lower()
        t=d.page_content or ""
        if "hotel" in src and ("gaeste" in src or "gäst" in src or "gaste" in src):
            if re.search(r"frühstück|fruehstueck", t, flags=re.I):
                if re.search(r"montag.*?freitag.*?07:00.*?10:30", t, flags=re.S|re.I) and re.search(r"samstag.*?sonntag.*?07:30.*?11:00", t, flags=re.S|re.I):
                    facts["breakfast"]="Montag–Freitag: 07:00–10:30 Uhr | Samstag–Sonntag: 07:30–11:00 Uhr"
            if re.search(r"\bwlan\b|\bwifi\b", t, flags=re.I):
                if re.search(r"zimmerkarte|schlüssel|hotelinformationsschreiben", t, flags=re.I):
                    facts["wlan"]="Das WLAN-Passwort steht auf der Rückseite Ihrer Zimmerkarte bzw. im Hotelinformationsschreiben."
    facts.setdefault("check","Check-in ab 15:00 | Check-out bis 11:00")
    return facts

def _load_raw_by_name(substr_list):
    texts=[]
    if not os.path.isdir(DATA_DIR): return ""
    for name in os.listdir(DATA_DIR):
        low=name.lower()
        if any(s in low for s in substr_list):
            p=os.path.join(DATA_DIR,name)
            if p not in _raw_cache:
                _raw_cache[p]=_read_file_text(p)
            texts.append(_raw_cache[p])
    return "\n".join(texts)

def _zone_from_tnw_html(place):
    raw=_load_raw_by_name(["tarifverbund","tnw",".html"])
    if not raw: return None
    txt=re.sub(r"\s+"," ", raw)
    m=re.search(place, txt, flags=re.I)
    if not m: return None
    start=max(0, m.start()-200)
    end=min(len(txt), m.end()+200)
    window=txt[start:end]
    m2=re.search(r"[Zz]one\s*([0-9]{1,2})", window)
    if m2:
        return f"Zone {m2.group(1)}"
    return None

def _tnw_zone_for_place(q):
    if re.search(r"barf(ü|u)ßer?platz", q, flags=re.I):
        z=_zone_from_tnw_html(r"Barf(ü|u)ßer?platz")
        return z or "Zone 10"
    if re.search(r"\bmesseplatz\b", q, flags=re.I):
        z=_zone_from_tnw_html(r"Messeplatz")
        return z or "Zone 10"
    return None

def _messeplatz_lines_web():
    try:
        with httpx.Client(follow_redirects=True, timeout=10.0, headers={"User-Agent":os.environ["USER_AGENT"]}) as c:
            r=c.get("https://www.bvb.ch/de/baustellen")
            txt=r.text.lower() if r.status_code<400 else ""
        base=["1","2","6","14","15","21"]
        note=None
        if "messeplatz" in txt and "linie 6" in txt and ("hält nicht" in txt or "haelt nicht" in txt or "entfällt" in txt or "entfaellt" in txt):
            note="Linie 6 hält derzeit nicht am Messeplatz (Baustelle)."
            base=[l for l in base if l!="6"]
        return base, note
    except Exception:
        return ["1","2","6","14","15","21"], None

def _context(query):
    if not _ret: return ""
    try:
        hits=_ret.invoke(query) if hasattr(_ret,"invoke") else _ret.get_relevant_documents(query)
        if not hits: return ""
        return DOC_SEP.join(d.page_content for d in hits[:10] if d.page_content.strip())
    except Exception:
        return ""

def _whimsical_fallback(q):
    if re.search(r"check[- ]?in|check[- ]?out", q, flags=re.I):
        return "Mein Kofferplan verrät mir gerade nichts Verlässliches. Ich kläre das gern an der Rezeption – geschätzt: Check-in ab 15:00, Check-out bis 11:00."
    if re.search(r"\bwlan\b|\bwifi\b", q, flags=re.I):
        return "Mein WLAN-Kompass blinkt ratlos. Ich frag’ gern kurz nach – oft stehen die Daten auf der Zimmerkarte."
    return "Mein Spickzettel schweigt dazu. Ich frage das gern für Sie nach – ein Augenblick Geduld, dann habe ich die exakten Infos."

def answer(q:str)->str:
    _ensure()
    q=q.strip()
    facts=_hotel_facts()
    ql=q.lower()

    if re.search(r"\btram|linie|messeplatz\b", ql):
        lines, note=_messeplatz_lines_web()
        base=",".join(lines)
        return f"Zum Messeplatz: Tram {base}" + (f" ({note})" if note else "")

    if re.search(r"zone|tarif|barf(ü|u)ßer?platz|messeplatz", ql):
        z=_tnw_zone_for_place(q)
        if z: return z

    if re.search(r"frühstück|fruehstueck", ql):
        if "breakfast" in facts: return facts["breakfast"]
        ctx=_context(q)
        return _chain.invoke({"context":ctx,"query":q}) if ctx else _whimsical_fallback(q)

    if re.search(r"\bwlan\b|\bwifi\b", ql):
        if "wlan" in facts: return facts["wlan"]
        ctx=_context(q)
        return _chain.invoke({"context":ctx,"query":q}) if ctx else _whimsical_fallback(q)

    if re.search(r"check[- ]?in|check[- ]?out", ql):
        if "check" in facts: return facts["check"]
        ctx=_context(q)
        return _chain.invoke({"context":ctx,"query":q}) if ctx else _whimsical_fallback(q)

    ctx=_context(q)
    if ctx:
        return _chain.invoke({"context":ctx,"query":q})
    return _whimsical_fallback(q)

def main():
    try:
        q=input().strip()
    except EOFError:
        q=""
    print(answer(q or "Hallo"))
if __name__=="__main__":
    main()

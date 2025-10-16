import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

env_file = find_dotenv(filename=".env", usecwd=True)
if not env_file:
    candidates = [
        Path(__file__).resolve().parents[1] / ".env",
        Path("/work/hotel-concierge-bot/.env"),
        Path("/datasets/_deepnote_work/hotel-concierge-bot/.env"),
    ]
    for p in candidates:
        if p.exists():
            env_file = str(p)
            break

if env_file:
    load_dotenv(env_file, override=True)

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-120b")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

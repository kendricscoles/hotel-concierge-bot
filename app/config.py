import os
from pathlib import Path
from dotenv import load_dotenv

possible_env_paths = [
    Path(__file__).resolve().parents[1] / ".env",
    Path("/work/hotel-concierge-bot/.env"),
    Path("/datasets/_deepnote_work/hotel-concierge-bot/.env"),
]

for env_path in possible_env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-120b")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

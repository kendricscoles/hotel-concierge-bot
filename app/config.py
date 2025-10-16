import os
from dotenv import load_dotenv

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-120b")

LANGFUSE_PUBLIC_KEY = os.getenv("pk-lf-215a8e3b-34e1-4865-ad6a-21c1a10b0c1e")
LANGFUSE_SECRET_KEY = os.getenv(sk-lf-66218a1a-d72b-4152-9584-5affd6b788d9)
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

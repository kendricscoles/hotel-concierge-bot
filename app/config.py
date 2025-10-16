import os
from dotenv import load_dotenv

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-120b")  

if not MODEL_NAME:
    raise RuntimeError("MODEL_NAME fehlt oder falsch konfiguriert.")

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

env_file = find_dotenv(filename=".env", usecwd=True)

if not env_file:
    possible_paths = [
        Path(__file__).resolve().parents[1] / ".env",
        Path("/work/hotel-concierge-bot/.env"),
        Path("/datasets/_deepnote_work/hotel-concierge-bot/.env"),
    ]
    for path in possible_paths:
        if path.exists():
            env_file = str(path)
            break

if env_file:
    load_dotenv(env_file, override=True)
    print(f"Loaded environment variables from: {env_file}")
else:
    print("No .env file found. Please create one with your API keys.")

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-120b")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

def show_config_summary():
    print("\nCONFIG SUMMARY")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"CEREBRAS_API_KEY present: {bool(CEREBRAS_API_KEY)}")
    if LANGFUSE_PUBLIC_KEY:
        print(f"Langfuse public: {LANGFUSE_PUBLIC_KEY[:10]}...")
    else:
        print("Langfuse public: None")
    print(f"Langfuse secret present: {bool(LANGFUSE_SECRET_KEY)}")
    print(f"Langfuse host: {LANGFUSE_HOST}")
    print("")

if __name__ == "__main__":
    show_config_summary()

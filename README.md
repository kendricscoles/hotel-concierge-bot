# hotel-concierge-bot
Hotel Concierge Chatbot using Langchain &amp; Cerebras


Run locally in PyCharm

Create a venv, then:
pip install -r requirements.txt
cp .env.example .env

Optionally put PDFs, HTML, or text files into ./data

Start the UI:
python -m app.ui

Open http://localhost:7860

LangSmith tracing is enabled through .env
RAG uses FAISS over ./data, with runtime upload support
If no relevant context is found, the bot asks the LLM directly

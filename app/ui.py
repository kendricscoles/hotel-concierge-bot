import os, pathlib, gradio as gr
from typing import List
from app.rag_basic import answer_with_llm

PORT = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))
HOST = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")

def _save_uploads(files: List[gr.File]) -> List[str]:
    if not files:
        return []
    up_dir = pathlib.Path("uploads")
    up_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        raw = open(f.name, "rb").read()
        dst = up_dir / pathlib.Path(getattr(f, "orig_name", pathlib.Path(f.name).name)).name
        with open(dst, "wb") as out:
            out.write(raw)
        paths.append(str(dst))
    return paths

def ask(q: str, files):
    paths = _save_uploads(files)
    ans, srcs = answer_with_llm(q, uploaded_paths=paths)
    return ans, srcs

with gr.Blocks(title="Hotel Concierge – Basel (RAG)") as demo:
    gr.Markdown("Hotel Concierge – Basel\nStelle eine Frage. Lade optional PDFs, HTML oder Text hoch.")
    q = gr.Textbox(label="Frage", placeholder="Wann ist der Check-in", lines=2)
    uploads = gr.File(label="Dokumente", file_count="multiple")
    btn = gr.Button("Antworten")
    a = gr.Markdown()
    s = gr.Markdown()
    btn.click(fn=ask, inputs=[q, uploads], outputs=[a, s])
    q.submit(fn=ask, inputs=[q, uploads], outputs=[a, s])

if __name__ == "__main__":
    demo.launch(server_name=HOST, server_port=PORT, share=False, show_error=True)

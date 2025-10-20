
import os, sys, pathlib, gradio as gr
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
import rag_basic

def ask(q: str):
    q = (q or "").strip()
    if not q:
        return "Bitte eine Frage eingeben."
    try:
        ans, src = rag_basic.answer_with_llm(q)
        if src:
            return f"{ans}\n\n_Quellen: {src}_"
        return ans
    except Exception as e:
        return f"Interner Fehler: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## Hotel Concierge – Basel")
    q = gr.Textbox(label="Frage", placeholder="z. B. Wann ist der Check-in?", lines=2)
    a = gr.Markdown()
    btn = gr.Button("Fragen")
    btn.click(fn=ask, inputs=q, outputs=a)
    q.submit(fn=ask, inputs=q, outputs=a)

# If your Gradio requires queue(), call it without args; otherwise you can omit it.
try:
    demo.queue()
except TypeError:
    pass

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True, show_api=False)

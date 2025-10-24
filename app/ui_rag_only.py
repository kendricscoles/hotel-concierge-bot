import gradio as gr
from app.rag_basic import answer_with_llm, trace_retrieve

def on_ask(q):
    rows = trace_retrieve(q, k=8)
    srcs = "\n".join([f"- {r['source']}  (score: {r['score']})" for r in rows]) or "- keine"
    ans = answer_with_llm(q)
    return ans, srcs

with gr.Blocks() as demo:
    gr.Markdown("Hotel Concierge RAG")
    q = gr.Textbox(label="Frage", placeholder="z. B. wann ist frühstück")
    a = gr.Markdown(label="Antwort")
    s = gr.Markdown(label="Gefundene Quellen")
    q.submit(on_ask, q, [a, s])
    gr.Button("Fragen").click(on_ask, q, [a, s])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
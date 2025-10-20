import os, gradio as gr
from app.rag_basic import answer_with_llm

PORT = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))
HOST = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")

def ask(q):
    return answer_with_llm(q)

with gr.Blocks(title="Hotel Concierge – Basel") as demo:
    gr.Markdown("Hotel Concierge – Basel\nStelle eine Frage.")
    q = gr.Textbox(label="Frage", placeholder="Wann ist der Check-in", lines=2)
    btn = gr.Button("Antworten")
    a = gr.Markdown()
    btn.click(fn=ask, inputs=[q], outputs=[a])
    q.submit(fn=ask, inputs=[q], outputs=[a])

if __name__ == "__main__":
    demo.launch(server_name=HOST, server_port=PORT, share=False, show_error=True)
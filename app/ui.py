import os
import gradio as gr
from app.rag_basic import answer_with_llm
from app.agent_tools_fallback import answer_with_tools

PORT = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860")))
HOST = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")

def ask(q):
    q = (q or "").strip()
    if not q:
        return "Gerne – womit darf ich Ihnen behilflich sein?"
    ans = ""
    try:
        ans = answer_with_llm(q)
    except Exception:
        ans = ""
    bad = (
        not ans
        or len(ans.strip()) < 5
        or ans.strip().lower().startswith(("leider", "ich habe keine informationen", "keine daten"))
        or "keine information" in ans.strip().lower()
        or "nicht gefunden" in ans.strip().lower()
    )
    if bad:
        try:
            ans2 = answer_with_tools(q)
        except Exception:
            ans2 = ""
        return ans2 or "Entschuldigung, das hat nicht geklappt."
    return ans

with gr.Blocks(title="Hotel Concierge – Basel") as demo:
    gr.Markdown("Hotel Concierge – Basel\nStelle eine Frage.")
    q = gr.Textbox(label="Frage", placeholder="Wann ist der Check-in", lines=2)
    btn = gr.Button("Antworten")
    a = gr.Markdown()
    btn.click(fn=ask, inputs=[q], outputs=[a])
    q.submit(fn=ask, inputs=[q], outputs=[a])

if __name__ == "__main__":
    demo.launch(server_name=HOST, server_port=PORT, share=False, show_error=True)
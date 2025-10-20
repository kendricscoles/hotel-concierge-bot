import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.rag_basic import answer
if len(sys.argv)>1:
    q=" ".join(sys.argv[1:])
else:
    q=input("Frage: ").strip()
print(answer(q))

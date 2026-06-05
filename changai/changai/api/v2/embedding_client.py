import requests
from langchain_core.embeddings import Embeddings
import frappe

@frappe.whitelist(allow_guest=False)
def get_local_embedding(text: str):
    try:
        res = requests.post(
            "http://127.0.0.1:8001/embed",
            json={"text": text},
            timeout=30
        )
        res.raise_for_status()
        return res.json()["embedding"]
    except Exception as e:
        frappe.throw(f"Embedding service error :{str(e)}")

class LocalEmbeddingService(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        return get_local_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [get_local_embedding(t) for t in texts]
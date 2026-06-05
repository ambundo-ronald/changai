from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
import os
MODEL_PATH = os.environ.get("CHANGAI_EMBEDDING_MODEL_PATH")

app = FastAPI()

embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={
        "device": "cpu",
        "trust_remote_code": True,
    },
    encode_kwargs={
        "normalize_embeddings": True,
    },
)

# warmup during service startup
embedding_model.embed_query("changai warmup")


class EmbedRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": True}


@app.post("/embed")
def embed(req: EmbedRequest):
    vector = embedding_model.embed_query(req.text)
    return {"embedding": vector}
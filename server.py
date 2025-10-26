import os

import clickhouse_connect
import pandas as pd
from anyio import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel
from qdrant_client import QdrantClient

from config import EMBEDDING_MODEL_PATH
from emcache import OllamaBackend
from emcache.huggingface import HuggingFaceBackend
from emseo.storage import VectorStoreEmbedding

load_dotenv()

ch_client = clickhouse_connect.get_client(
    host=os.getenv("CLICKHOUSE_HOST"),
    port=int(os.getenv("CLICKHOUSE_PORT")),
    username=os.getenv("CLICKHOUSE_USERNAME"),
    password=os.getenv("CLICKHOUSE_PASSWORD"),
)


client = QdrantClient(url="http://localhost:6333")

# backend = OllamaBackend(base_url="http://127.0.0.1:11434")
# backend = HuggingFaceBackend(model_name="heydariAI/persian-embeddings")
backend = HuggingFaceBackend(model_name="intfloat/multilingual-e5-large")
storage = VectorStoreEmbedding(backend, collection_prefix="keywords")

app = FastAPI(title="Website Semantic Search API")


class SearchResult(BaseModel):
    website: str
    score: float


@app.get("/search", response_model=list[SearchResult])
def search(query: str = Query(...), top_k: int = 32):
    """Search websites similar to the input query."""

    # Query the vector keyword storage
    results = storage.search(query=query, top_k=top_k)

    kw_website_pairs = []
    websites = []
    for r in results.points:
        if "keyword" in r.payload:
            kw_website_pairs.append(
                {
                    "similarity": r.score,
                    "average_position": r.payload["average_position"],
                    "website": r.payload["website"],
                    "keyword": r.payload["keyword"],
                }
            )

        else:
            ...  # TODO: warn

    kw_website_pairs = pd.DataFrame(kw_website_pairs)
    websites = pd.DataFrame(websites)

    # Query ClickHouse to get rankings
    query_ranks = Path("queries/website_by_keyword.sql").read_text(encoding="utf-8")

    ranks = ch_client.query_df(
        query_ranks,
        parameters={
            "keywords": kw_website_pairs["keyword"].tolist(),
            "similarity": kw_website_pairs["similarity"].tolist(),
        },
    ).rename(columns={"q.similarity": "similarity"})

    # Compute mean scores

    ranks["score"] = ranks["similarity"] / ranks["average_position"]
    ranked_websites: pd.DataFrame = (
        ranks.groupby("website")["score"].mean().sort_values(ascending=False)
    )

    return ranked_websites.to_records(index=False).tolist()

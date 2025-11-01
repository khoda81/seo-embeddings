import logging
import os
from pathlib import Path

import clickhouse_connect
import dotenv
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel

from emcache import OllamaBackend
from emcache.huggingface import HuggingFaceBackend
from emseo.storage import VectorStoreEmbedding

dotenv.load_dotenv()


# ------------------------------
# Configuration
# ------------------------------
BACKEND = os.getenv("BACKEND", "huggingface").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-large")
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Initialize Databases
logger.info("Connecting to ClickHouse...")
ch_client = clickhouse_connect.get_client(
    host=os.getenv("CLICKHOUSE_HOST"),
    port=int(os.getenv("CLICKHOUSE_PORT")),
    username=os.getenv("CLICKHOUSE_USERNAME"),
    password=os.getenv("CLICKHOUSE_PASSWORD"),
)

logger.info(
    "Connected to ClickHouse at %s:%s",
    os.getenv("CLICKHOUSE_HOST"),
    os.getenv("CLICKHOUSE_PORT"),
)

# logger.info("Testing clickhouse with a simple query...")
# query_ranks = Path("queries/website_by_keyword.sql").read_text(encoding="utf-8")
# df = ch_client.query_df(
#     query_ranks, parameters={"keywords": ["hi", "buy"], "similarity": [0.9, 0.8]}
# )
# logger.info("Test query result:\n%s", df.head())

# Select Backend
if BACKEND == "ollama":
    model_name = MODEL_NAME or "nomic-text-embed"
    backend = OllamaBackend(model=model_name, base_url="http://127.0.0.1:11434")
    logger.info("Using Ollama backend with model '%s'", model_name)

elif BACKEND == "huggingface":
    model_name = MODEL_NAME or "intfloat/multilingual-e5-large"
    backend = HuggingFaceBackend(model_name=model_name)
    logger.info("Using HuggingFace backend with model '%s'", model_name)

else:
    raise ValueError(f"Unrecognized backend {BACKEND}")


storage = VectorStoreEmbedding(
    backend,
    collection_prefix="keywords",
    qdrant_url=QDRANT_URL,
)
logger.info("VectorStoreEmbedding initialized successfully.")


# FastAPI Setup
app = FastAPI(title="Website Semantic Search API")


class SearchResult(BaseModel):
    website: str
    score: float


@app.get("/search", response_model=list[SearchResult])
def search(query: str = Query(...), top_k: int = 32):
    """Search websites similar to the input query."""
    logger.info("Received search query='%s' (top_k=%d)", query, top_k)

    results = storage.search(query=query, top_k=top_k)
    logger.info("Retrieved %d results from vector store", len(results.points))

    kw_website_pairs = []
    for r in results.points:
        if "text" in r.payload:
            kw_website_pairs.append(
                {
                    "similarity": r.score,
                    "keyword": r.payload["text"],
                }
            )

        else:
            logger.warning("Missing 'text' in payload: %s", r.payload)

    kw_website_pairs = pd.DataFrame(kw_website_pairs)
    query_ranks = Path("queries/website_by_keyword.sql").read_text(encoding="utf-8")
    logger.debug("Query: %s", query_ranks)
    logger.debug("Parameters: %s", kw_website_pairs.head())

    ranks = ch_client.query_df(
        query_ranks,
        parameters={
            "keywords": kw_website_pairs["keyword"].tolist(),
            "similarity": kw_website_pairs["similarity"].tolist(),
        },
    ).rename(columns={"q.similarity": "similarity"})

    ranks["score"] = ranks["similarity"] / ranks["average_position"]
    ranked_websites = (
        ranks.groupby("website")["score"].mean().sort_values(ascending=False)
    )

    logger.info("Returning %d ranked websites", len(ranked_websites))

    # Convert the Series → list of dicts
    result = [
        {"website": website, "score": float(score)}
        for website, score in ranked_websites.items()
    ]

    logger.debug("Results: %s", result)
    return result


class KeywordResult(BaseModel):
    keyword: str
    similarity: float


@app.get("/similar-keywords", response_model=list[KeywordResult])
def similar_keywords(query: str = Query(...), top_k: int = 32):
    """Search keywords similar to the input query."""

    results = storage.search(query=query, top_k=top_k)
    logger.info("Retrieved %d results from vector store", len(results.points))

    kw_website_pairs = []
    for r in results.points:
        if "text" in r.payload:
            kw_website_pairs.append(
                {
                    "similarity": r.score,
                    "keyword": r.payload["text"],
                }
            )

        else:
            logger.warning("Missing 'text' in payload: %s", r.payload)

    return kw_website_pairs


class RankEntry(BaseModel):
    keyword: str
    website: str
    average_position: float
    similarity: float


@app.get("/ranking-entries", response_model=list[RankEntry])
def entries(query: str = Query(...), top_k: int = 32):
    """Search keywords similar to the input query."""

    results = storage.search(query=query, top_k=top_k)
    logger.info("Retrieved %d results from vector store", len(results.points))

    kw_website_pairs = []
    for r in results.points:
        if "text" in r.payload:
            kw_website_pairs.append(
                {
                    "similarity": r.score,
                    "keyword": r.payload["text"],
                }
            )

        else:
            logger.warning("Missing 'text' in payload: %s", r.payload)

    kw_website_pairs = pd.DataFrame(kw_website_pairs)
    query_ranks = Path("queries/website_by_keyword.sql").read_text(encoding="utf-8")
    logger.debug("Query: %s", query_ranks)
    logger.debug("Parameters: %s", kw_website_pairs.head())

    ranks = ch_client.query_df(
        query_ranks,
        parameters={
            "keywords": kw_website_pairs["keyword"].tolist(),
            "similarity": kw_website_pairs["similarity"].tolist(),
        },
    ).rename(columns={"q.similarity": "similarity"})

    return ranks.to_dict(orient="records")


if __name__ == "__main__":
    logger.info("Starting API...")
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

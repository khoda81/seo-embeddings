import logging
import os
from collections import defaultdict
from enum import Enum
from pathlib import Path

import clickhouse_connect
import dotenv
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

from emcache import OllamaBackend
from emcache.huggingface import HuggingFaceBackend
from emseo.backend.agent import generate_similar_terms
from emseo.storage import VectorStoreEmbedding

dotenv.load_dotenv()


# ------------------------------
# Configuration
# ------------------------------
BACKEND = os.getenv("BACKEND", "huggingface").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-large")
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
OPENROUTER_MODEL = "gpt-4o-mini"


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


# Select Backend
if BACKEND == "ollama":
    model_name = MODEL_NAME or "nomic-text-embed"
    backend = OllamaBackend(model=model_name, base_url="http://127.0.0.1:11434")
    logger.info("Using Ollama backend with model '%s'", model_name)

elif BACKEND == "huggingface":
    model_name = MODEL_NAME or "intfloat/multilingual-e5-large"
    backend = HuggingFaceBackend(model_name=model_name, trust_remote_code=True)
    logger.info("Using HuggingFace backend with model '%s'", model_name)

else:
    raise ValueError(f"Unrecognized backend {BACKEND}")


storage = VectorStoreEmbedding(
    backend,
    collection_prefix="keywords",
    qdrant_url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
logger.info("VectorStoreEmbedding initialized successfully.")


openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
)

# FastAPI Setup
app = FastAPI(title="Topical Authority Keyword Similarity API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: specify domain list
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchResult(BaseModel):
    website: str
    score: float


class ScoreFunction(str, Enum):
    WEIGHTED_SIM = "weighted_sim"
    LOG_MEAN = "log_mean"


@app.get("/search", response_model=list[SearchResult])
def search(
    query: str = Query(..., description="Search query string"),
    top_k: int = Query(32, ge=1, description="Top-k keyword limit"),
    score_function: ScoreFunction = Query(ScoreFunction.LOG_MEAN),
):
    """Search websites similar to the input query."""
    logger.info("Received search query='%s' (top_k=%d)", query, top_k)

    # --- Query the vector store ---
    results = storage.search(query=query, top_k=top_k)
    logger.info("Retrieved %d results from vector store", len(results.points))

    # --- Build keyword–similarity pairs ---
    kw_website_pairs = []
    for r in results.points:
        payload = r.payload or {}
        if "text" in payload:
            kw_website_pairs.append(
                {
                    "similarity": r.score,
                    "keyword": payload["text"],
                }
            )
        else:
            logger.warning("Missing 'text' in payload: %s", payload)

    if not kw_website_pairs:
        logger.warning("No valid keyword payloads found — returning empty result.")
        return []

    kw_website_pairs = pd.DataFrame(kw_website_pairs)

    # --- Query ranks ---
    query_ranks = Path("queries/website_by_keyword.sql").read_text(encoding="utf-8")
    ranks = ch_client.query_df(
        query_ranks,
        parameters={
            "keywords": kw_website_pairs["keyword"].tolist(),
            "similarity": kw_website_pairs["similarity"].tolist(),
        },
    ).rename(columns={"q.similarity": "similarity"})

    # --- Compute scores depending on function ---
    if score_function == ScoreFunction.WEIGHTED_SIM:
        ranks["score"] = ranks["similarity"] / ranks["average_position"]

        ranked_websites = (
            ranks.groupby("website")["score"].mean().sort_values(ascending=False)
        )

    elif score_function == ScoreFunction.LOG_MEAN:
        # Guard: make sure `score` exists
        if "score" not in ranks.columns:
            ranks["score"] = ranks["similarity"] / ranks["average_position"]

        agg = ranks.groupby("website").agg(
            mean_score=("score", "mean"),
            n=("score", "count"),
        )
        agg["final_score"] = agg["mean_score"] * (1 + np.log1p(agg["n"]))
        ranked_websites = agg["final_score"].sort_values(ascending=False)

    else:
        # TODO: Return a proper error response
        logger.error("Unsupported score_function: %s", score_function)
        return []

    # --- Prepare and return response ---
    result = [
        SearchResult(website=str(website), score=float(score))
        for website, score in ranked_websites.items()
    ]

    logger.info("Returning %d ranked websites", len(result))
    return result


class KeywordResult(BaseModel):
    keyword: str
    similarity: float


@app.get("/similar-keywords", response_model=list[KeywordResult])
def similar_keywords(
    query: str = Query(...),
    top_k: int = Query(32, ge=1, description="Top-k keyword limit"),
    augmentation: bool = False,
    similarity_threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    """Search keywords similar to the input query, with optional augmentation and similarity threshold."""

    # -----------------------------
    # AUGMENTATION MODE
    # -----------------------------
    if augmentation:
        queries = generate_similar_terms(
            term=query,
            model=OPENROUTER_MODEL,
            openai_client=openai_client,
        )

        # pass threshold to batched search
        results = storage.search_batched(
            queries=queries,
            top_k=top_k,
            score_threshold=similarity_threshold,
        )

        scores = defaultdict(float)
        for result in results:
            for r in result.points:
                if "text" in r.payload:
                    scores[r.payload["text"]] += r.score
                else:
                    logger.warning("Missing 'text' in payload: %s", r.payload)

        n = len(queries)
        output = [
            KeywordResult(keyword=kw, similarity=sc / n) for (kw, sc) in scores.items()
        ]

        output.sort(key=lambda r: r.similarity, reverse=True)
        return output

    # -----------------------------
    # NORMAL MODE
    # -----------------------------
    results = storage.search(
        query=query,
        top_k=top_k,
        score_threshold=similarity_threshold,
        timeout=60,
    )

    logger.info("Retrieved %d results from vector store", len(results.points))

    output = []
    for r in results.points:
        output.append(
            KeywordResult(
                similarity=r.score,
                keyword=r.payload["text"],
            )
        )

    return output


class RankEntry(BaseModel):
    keyword: str
    website: str
    average_position: float
    similarity: float


@app.get("/ranking-entries", response_model=list[RankEntry])
def entries(
    query: str = Query(...),
    top_k: int = Query(32, ge=1, description="Top-k keyword limit"),
):
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

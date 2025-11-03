import logging

from qdrant_client import QdrantClient
from qdrant_client.http.models import CollectionInfo, QueryResponse
from qdrant_client.models import Distance, PointStruct, QueryRequest, VectorParams

from emcache.base import BaseCachedEmbedding

logger = logging.getLogger(__name__)


class VectorStoreEmbedding:
    def __init__(
        self,
        backend,
        collection_prefix: str,
        qdrant_url: str = "http://127.0.0.1:6333",
        distance: Distance = Distance.COSINE,
    ):
        self.backend = backend
        self.embedder = BaseCachedEmbedding(backend)
        self.qdrant = QdrantClient(url=qdrant_url)

        # Unique collection name per model/backend
        self.collection_name = f"{collection_prefix}_{self.backend.identifier()}"

        # Create collection if needed
        self._ensure_collection(distance)

    def _ensure_collection(self, distance: Distance):
        if self.qdrant.collection_exists(self.collection_name):
            points = self.info().points_count

            logger.info(
                f"🧩 Using existing Qdrant collection: {self.collection_name} with {points} points"
            )
            return

        logger.info(f"🧩 Creating Qdrant collection: {self.collection_name}")
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedder.embed_dim,
                distance=distance,
            ),
        )

    def add_texts(
        self,
        texts: list[str],
        payloads: list[dict] | None = None,
    ) -> list[str]:
        """Embeds and inserts new texts into Qdrant."""
        if payloads is None:
            payloads = [{} for _ in texts]

        embed_results = self.embedder.embed(texts)
        embeddings = embed_results.embedding.cpu().tolist()
        ids = embed_results.text_ids.cpu().tolist()

        points = [
            PointStruct(id=id_, vector=emb, payload={"text": text, **payload})
            for id_, emb, text, payload in zip(ids, embeddings, texts, payloads)
        ]

        self.add_points(points)
        return texts

    def add_points(self, points: list[PointStruct]) -> None:
        """Inserts pre-embedded points into Qdrant."""
        self.qdrant.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query: str | list[float],
        top_k: int = 5,
        with_payload: bool = True,
        score_threshold: float | None = None,
        cached: bool = False,
    ) -> QueryResponse:
        if isinstance(query, str):
            if cached:
                vector = self.embedder.embed([query]).embedding[0].cpu().tolist()

            else:
                vector = self.embedder.backend.embed_texts([query])[0]

        elif isinstance(query, list):
            vector = query

        else:
            raise ValueError("Query must be a string or a list of floats.")

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            with_payload=with_payload,
            score_threshold=score_threshold,
        )

        return results

    def search_batched(
        self,
        queries: list[str] | list[list[float]],
        top_k: int = 5,
        with_payload: bool = True,
        score_threshold: float | None = None,
        cached: bool = False,
    ) -> list[QueryResponse]:
        """Performs a batched search for multiple queries."""
        if isinstance(queries[0], str):
            if cached:
                vectors = self.embedder.embed(queries).embedding.cpu().tolist()

            else:
                vectors = self.embedder.backend.embed_texts(queries)

        elif isinstance(queries, list):
            vectors = queries

        else:
            raise ValueError("Query must be a string or a list of floats.")

        results = self.qdrant.query_batch_points(
            collection_name=self.collection_name,
            requests=[
                QueryRequest(
                    query=v,
                    limit=top_k,
                    with_payload=with_payload,
                    score_threshold=score_threshold,
                )
                for v in vectors
            ],
        )

        return results

    def info(self) -> CollectionInfo:
        return self.qdrant.get_collection(self.collection_name)

    def clear(self):
        """Deletes all points in the collection."""
        self.qdrant.delete(
            collection_name=self.collection_name, points_selector={"filter": {}}
        )

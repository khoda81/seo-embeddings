from sentence_transformers import SentenceTransformer


class HuggingFaceBackend:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def get_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def identifier(self) -> str:
        backend_name = self.__class__.__name__.replace("Backend", "").lower()
        model_name = self.model_name.replace("/", "_")
        return f"{backend_name}_{model_name}"

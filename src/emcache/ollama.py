import requests


class OllamaBackend:
    def __init__(self, model="nomic-embed-text", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def get_dim(self) -> int:
        res = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": "test"},
        )
        res.raise_for_status()
        return len(res.json()["embedding"])

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        res = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        res.raise_for_status()
        return res.json()["embeddings"]

    def identifier(self) -> str:
        backend_name = self.__class__.__name__.replace("Backend", "").lower()
        model_name = self.model.replace("/", "_")
        return f"{backend_name}_{model_name}"

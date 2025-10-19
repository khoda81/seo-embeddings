import requests
import torch
from torch import nn


class OllamaEmbedding(nn.Module):
    embed_cache: nn.Buffer
    embed_keys: dict[str, int]

    def __init__(self, model="nomic-embed-text"):
        super().__init__()
        self.model = model

        self.embed_dim = self._get_text_dim()
        self.register_buffer("embed_cache", torch.empty(0, self.embed_dim))
        self.embed_keys = {}

    def state_dict(self, *args, **kwargs) -> dict[str]:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["embed_keys"] = self.embed_keys
        return state_dict

    def load_state_dict(self, state_dict: dict, strict=True, assign=False):
        self.embed_keys = state_dict.pop("embed_keys")
        self.embed_cache = state_dict["embed_cache"]

        return super().load_state_dict(state_dict, strict, assign)

    def _get_text_dim(self) -> int:
        res = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": self.model, "prompt": "test"},
        )
        res.raise_for_status()

        return len(res.json()["embedding"])

    def text_id(self, text: str):
        new_id = len(self.embed_keys)
        return self.embed_keys.setdefault(text, new_id)

    def embed(self, texts: list[str]) -> torch.Tensor:
        new_texts = [text for text in texts if text not in self.embed_keys]
        if new_texts:
            res = requests.post(
                "http://localhost:11434/api/embed",
                json={"model": self.model, "input": new_texts},
            )

            res.raise_for_status()
            new_embs = torch.tensor(
                res.json()["embeddings"],
                device=self.embed_cache.device,
                dtype=self.embed_cache.dtype,
            )

            all_embeds = [self.embed_cache.data, new_embs]

            size, dim = self.embed_cache.shape
            self.embed_cache.data = torch.empty(
                len(new_embs) + size,
                dim,
                device=self.embed_cache.device,
                dtype=self.embed_cache.dtype,
            )

            torch.cat(all_embeds, dim=0, out=self.embed_cache.data)

        indices = torch.tensor(
            [self.text_id(text) for text in texts],
            dtype=torch.long,
            device=self.embed_cache.device,
        )

        return self.embed_cache[indices]

    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.embed(texts)

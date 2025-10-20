from typing import Protocol, runtime_checkable

import attrs
import torch
from torch import nn


@runtime_checkable
class EmbeddingBackend(Protocol):
    def get_dim(self) -> int:
        """Return the embedding dimension."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of texts."""
        ...

    def identifier(self) -> str:
        """Return a unique name for this backend+model combination."""
        ...


@attrs.define
class EmbedResult:
    text_ids: torch.LongTensor
    embedding: torch.Tensor


class BaseCachedEmbedding(nn.Module):
    embed_cache: nn.Buffer
    embed_keys: dict[str, int]

    def __init__(self, backend: EmbeddingBackend, dtype=torch.float32):
        super().__init__()
        self.backend = backend
        self.embed_dim = backend.get_dim()
        self.register_buffer("embed_cache", torch.empty(0, self.embed_dim, dtype=dtype))
        self.embed_keys = {}
        self.embed_ids = []

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["embed_keys"] = self.embed_keys
        state_dict["embed_ids"] = self.embed_ids
        return state_dict

    def load_state_dict(self, state_dict: dict, strict=True, assign=False):
        self.embed_keys = state_dict.pop("embed_keys")
        self.embed_ids = state_dict["embed_ids"]
        return super().load_state_dict(state_dict, strict, assign)

    def text_id(self, text: str) -> int:
        new_id = len(self.embed_keys)
        self.embed_ids.append(new_id)
        return self.embed_keys.setdefault(text, new_id)

    def embed(self, texts: list[str]) -> EmbedResult:
        new_texts = [t for t in texts if t not in self.embed_keys]
        if new_texts:
            new_embs = torch.tensor(
                self.backend.embed_texts(new_texts),
                device=self.embed_cache.device,
                dtype=self.embed_cache.dtype,
            )

            # expand cache
            self.embed_cache.data = torch.cat([self.embed_cache.data, new_embs], dim=0)

        indices = torch.tensor(
            [self.text_id(t) for t in texts],
            dtype=torch.long,
            device=self.embed_cache.device,
        )

        return EmbedResult(
            text_ids=indices,
            embedding=self.embed_cache[indices],
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.embed(texts)

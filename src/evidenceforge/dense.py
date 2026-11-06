from __future__ import annotations

from pathlib import Path
from threading import Lock

from .models import Document

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class E5Encoder:
    """E5 with the publisher's masked mean pooling and normalized vectors."""

    def __init__(self, model_path: str | Path | None = None, threads: int = 6):
        try:
            import numpy as np
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install evidenceforge[dense] to use E5") from error

        if not 1 <= threads <= 32:
            raise ValueError("E5 threads must be between 1 and 32")
        options = {"revision": MODEL_REVISION} if model_path is None else {"local_files_only": True}
        model = MODEL_ID if model_path is None else str(Path(model_path).resolve(strict=True))
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(model, **options)
        self.model = AutoModel.from_pretrained(model, use_safetensors=True, **options).eval()
        self.torch = torch
        self.numpy = np
        self.lock = Lock()

    def encode(self, texts: list[str]):
        torch = self.torch
        output = self.numpy.empty((len(texts), self.model.config.hidden_size), dtype="float32")
        order = sorted(range(len(texts)), key=lambda index: -len(texts[index]))
        with self.lock, torch.inference_mode():
            for start in range(0, len(order), 32):
                batch = order[start : start + 32]
                inputs = self.tokenizer(
                    [texts[index] for index in batch],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                states = self.model(**inputs).last_hidden_state
                mask = inputs.attention_mask[..., None]
                pooled = states.masked_fill(~mask.bool(), 0).sum(dim=1) / mask.sum(dim=1)
                output[batch] = torch.nn.functional.normalize(pooled, dim=1).numpy()
        return output

    def encode_passages(self, docs: tuple[Document, ...]):
        return self.encode([f"passage: {doc.title} {doc.text}" for doc in docs])

    def encode_query(self, query: str):
        return self.encode([f"query: {query}"])[0]

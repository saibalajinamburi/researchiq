import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MAX_LENGTH = int(os.getenv("RESEARCHIQ_MAX_LENGTH", "192"))
DEFAULT_CACHE_SIZE = int(os.getenv("RESEARCHIQ_CACHE_SIZE", "512"))


def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask_expanded = np.expand_dims(attention_mask, -1)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


class TTLCache:
    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE, ttl_seconds: int = 3600) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            created_at, value = item
            if now - created_at > self.ttl_seconds:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._items[key] = (time.time(), value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"size": len(self._items), "max_size": self.max_size}


class ResearchIQInference:
    def __init__(
        self,
        classifier_path: Path | None = None,
        metadata_path: Path | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.classifier_path = classifier_path or ROOT_DIR / "models" / "phase4_onnx" / "best_model.onnx"
        self.metadata_path = metadata_path or ROOT_DIR / "models" / "phase3_final" / "best_model_metadata.json"
        self.max_length = max_length
        self.cache = TTLCache()
        self._lock = Lock()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            tokenizer_path = hf_hub_download(
                repo_id="Xenova/all-MiniLM-L6-v2",
                filename="tokenizer.json",
            )
            embedder_path = hf_hub_download(
                repo_id="Xenova/all-MiniLM-L6-v2",
                filename="onnx/model_quantized.onnx",
            )

            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.tokenizer.enable_truncation(max_length=self.max_length)
            self.tokenizer.enable_padding(length=self.max_length)

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = max(1, min(8, os.cpu_count() or 4))
            sess_options.inter_op_num_threads = 1

            self.embedder_session = ort.InferenceSession(
                embedder_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self.classifier_session = ort.InferenceSession(
                str(self.classifier_path),
                providers=["CPUExecutionProvider"],
            )
            self.classifier_input_name = self.classifier_session.get_inputs()[0].name

            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.classes = metadata["classes"]
            self.model_metadata = metadata
            self._loaded = True

    def embed(self, texts: list[str]) -> np.ndarray:
        self.load()
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([item.ids for item in encodings], dtype=np.int64)
        attention_mask = np.array([item.attention_mask for item in encodings], dtype=np.int64)
        token_type_ids = np.array([item.type_ids for item in encodings], dtype=np.int64)

        token_embeddings = self.embedder_session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )[0]
        return mean_pooling(token_embeddings, attention_mask).astype(np.float32)

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        self.load()
        uncached_indexes: list[int] = []
        uncached_texts: list[str] = []
        results: list[dict[str, Any] | None] = [None] * len(texts)

        for index, text in enumerate(texts):
            key = self.cache_key(text)
            cached = self.cache.get(key)
            if cached is not None:
                cached = {**cached, "cached": True}
                results[index] = cached
            else:
                uncached_indexes.append(index)
                uncached_texts.append(text)

        if uncached_texts:
            embeddings = self.embed(uncached_texts)
            labels, probabilities = self.classifier_session.run(
                None,
                {self.classifier_input_name: embeddings},
            )
            labels = np.asarray(labels).astype(int)
            probabilities = np.asarray(probabilities)

            for local_index, original_index in enumerate(uncached_indexes):
                label_id = int(labels[local_index])
                class_probs = probabilities[local_index]
                confidence = float(np.max(class_probs))
                top_indexes = np.argsort(class_probs)[::-1][:3]
                item = {
                    "label_id": label_id,
                    "category": self.classes[label_id],
                    "confidence": confidence,
                    "top_categories": [
                        {
                            "label_id": int(idx),
                            "category": self.classes[int(idx)],
                            "probability": float(class_probs[int(idx)]),
                        }
                        for idx in top_indexes
                    ],
                    "cached": False,
                }
                self.cache.set(self.cache_key(texts[original_index]), item)
                results[original_index] = item

        return [item for item in results if item is not None]

    def predict(self, text: str) -> dict[str, Any]:
        return self.predict_batch([text])[0]

    def cache_key(self, text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    def info(self) -> dict[str, Any]:
        self.load()
        return {
            "classifier_path": str(self.classifier_path),
            "embedding_model": "Xenova/all-MiniLM-L6-v2:onnx/model_quantized.onnx",
            "classes": self.classes,
            "metadata": self.model_metadata,
            "cache": self.cache.stats(),
            "providers": ort.get_available_providers(),
        }

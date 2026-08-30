"""Yandex embeddings client (text-search-doc / text-search-query) with disk cache."""

import hashlib
import json
import os
import time

import numpy as np
import requests

EMBED_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"


class EmbeddingsClient:
    """Embeds documents and queries via Yandex Foundation Models Embeddings API.

    Document vectors are cached on disk (npz) keyed by chunk id so the index
    is built once and reused by every benchmark run.
    """

    def __init__(self, api_key, folder_id, cache_path=None, rps=7, retries=5, query_cache_path=None):
        self.api_key = api_key
        self.folder_id = folder_id
        self.cache_path = cache_path
        self.query_cache_path = query_cache_path
        self.doc_uri = "emb://{}/text-search-doc/latest".format(folder_id)
        self.query_uri = "emb://{}/text-search-query/latest".format(folder_id)
        self.min_interval = 1.0 / rps
        self._last_call = 0.0
        self.retries = retries
        self._cache = self._load_cache()
        self._qcache = self._load_qcache()

    @property
    def cache(self):
        return self._cache

    def _load_cache(self):
        if self.cache_path and os.path.exists(self.cache_path):
            data = np.load(self.cache_path, allow_pickle=True)
            ids = data["ids"].tolist()
            vecs = data["vecs"]
            return {
                int(cid): vec.astype(np.float32)
                for cid, vec in zip(ids, vecs)
            }
        return {}

    def save_cache(self):
        """Persist accumulated document vectors to the npz cache."""
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        ids = sorted(self._cache)
        vecs = np.vstack([self._cache[cid] for cid in ids])
        np.savez_compressed(self.cache_path, ids=np.array(ids), vecs=vecs)

    def _throttle(self):
        wait = self._last_call + self.min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _request(self, uri, text):
        body = {"modelUri": uri, "text": text[:8000]}
        for attempt in range(self.retries + 1):
            self._throttle()
            resp = requests.post(
                EMBED_URL,
                json=body,
                headers={"Authorization": "Api-Key " + self.api_key},
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()["embedding"]
            if resp.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(
                "Embeddings API error {}: {}".format(resp.status_code, resp.text[:300])
            )
        raise RuntimeError("Embeddings API rate limit exceeded after retries")

    def embed_doc(self, chunk_id, text):
        """Return (and cache) the doc-vector for a chunk id."""
        if chunk_id in self._cache:
            return self._cache[chunk_id]
        vec = self._request(self.doc_uri, text)
        self._cache[chunk_id] = np.array(vec, dtype=np.float32)
        return self._cache[chunk_id]

    def _load_qcache(self):
        if self.query_cache_path and os.path.exists(self.query_cache_path):
            with open(self.query_cache_path, encoding="utf-8") as fh:
                data = json.load(fh)
            return {k: np.array(v, dtype=np.float32) for k, v in data.items()}
        return {}

    def _save_qcache(self):
        if not self.query_cache_path:
            return
        os.makedirs(os.path.dirname(self.query_cache_path), exist_ok=True)
        with open(self.query_cache_path, "w", encoding="utf-8") as fh:
            json.dump({k: v.tolist() for k, v in self._qcache.items()}, fh)

    def embed_query(self, text):
        """Return the query-vector via text-search-query, cached by text hash."""
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in self._qcache:
            return self._qcache[key]
        vec = np.array(self._request(self.query_uri, text), dtype=np.float32)
        self._qcache[key] = vec
        self._save_qcache()
        return vec

    def doc_vectors(self, chunk_ids):
        """Return a stacked matrix aligned with chunk_ids."""
        missing = [cid for cid in chunk_ids if cid not in self._cache]
        if missing:
            raise RuntimeError(
                "Missing document vectors for {} chunk ids".format(len(missing))
            )
        return np.vstack([self._cache[cid] for cid in chunk_ids])
#Developed by YaAgent / SagaAI Platform, 2026. https://github.com/TVD2100/sagaai-platform
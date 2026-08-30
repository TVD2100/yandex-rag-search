"""Vector search retriever over the YaAgentAI chunk base.

The retriever embeds queries with text-search-query and scores document
chunks by cosine similarity against the document embedding matrix. Document
vectors are loaded either from the on-disk npz cache (small bases) or
straight from the SQLite embeddings table (large bases, use_db_vectors=True).
"""

import numpy as np

from .db import load_chunks, load_vectors
from .embeddings import EmbeddingsClient


def normalize(vec):
    """L2-normalize a vector."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class Retriever:
    """Vector-only retriever over a chunk base.

    Parameters
    ----------
    db_path : str
        Path to the SQLite index (chunks + embeddings tables).
    api_key, folder_id : str
        Yandex Cloud credentials for the embeddings API.
    cache_path : str, optional
        Path to the npz document-vector cache (used when use_db_vectors=False).
    query_cache_path : str, optional
        Path to the JSON query-vector cache.
    rps : int
        Requests per second limit for the embeddings API.
    vec_k : int
        Default number of chunks returned by search_vector.
    use_db_vectors : bool
        When True, load document vectors from the SQLite embeddings table
        instead of the npz cache (required for large bases without a cache).
    """

    def __init__(
        self,
        db_path,
        api_key,
        folder_id,
        cache_path=None,
        query_cache_path=None,
        rps=7,
        vec_k=10,
        use_db_vectors=False,
    ):
        self.db_path = db_path
        self.chunks, self.by_id = load_chunks(db_path)
        self.client = EmbeddingsClient(
            api_key, folder_id, cache_path, rps=rps, query_cache_path=query_cache_path
        )
        self.api_key = api_key
        self.folder_id = folder_id
        self.vec_k = vec_k
        self.use_db_vectors = use_db_vectors
        self.ids_order = [c["id"] for c in self.chunks]
        self._doc_matrix = None
        if use_db_vectors:
            self._load_db_vectors()

    def _load_db_vectors(self):
        """Load document vectors from the SQLite embeddings table."""
        db_ids, matrix = load_vectors(self.db_path)
        if db_ids == self.ids_order:
            self._doc_matrix = normalize(matrix)
            return
        by_id = {cid: vec for cid, vec in zip(db_ids, matrix)}
        aligned = np.vstack([by_id[cid] for cid in self.ids_order])
        self._doc_matrix = normalize(aligned)

    def build_index(self, verbose=False):
        """Embed all chunks with text-search-doc (cached) and persist."""
        total = len(self.chunks)
        for idx, chunk in enumerate(self.chunks):
            self.client.embed_doc(chunk["id"], chunk["text"])
            if verbose and (idx + 1) % 200 == 0:
                print("embedded {}/{}".format(idx + 1, total))
                self.client.save_cache()
        self.client.save_cache()
        self._doc_matrix = normalize(self.client.doc_vectors(self.ids_order))

    @property
    def doc_matrix(self):
        """Normalized document embedding matrix; built on first access."""
        if self._doc_matrix is None:
            self._doc_matrix = normalize(self.client.doc_vectors(self.ids_order))
        return self._doc_matrix

    def search_vector(self, query, top_n=None):
        """Pure vector search via text-search-query: [(chunk_id, score)]."""
        q = normalize(self.client.embed_query(query))
        sims = self.doc_matrix @ q
        order = np.argsort(-sims)[: top_n or self.vec_k]
        return [(self.ids_order[i], float(sims[i])) for i in order]
#Developed by YaAgent / SagaAI Platform, 2026. https://github.com/TVD2100/sagaai-platform
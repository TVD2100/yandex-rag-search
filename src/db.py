"""SQLite access for the YaAgentAI chunk index with metadata-prefix cleaning."""

import re
import sqlite3

import numpy as np

_PREFIX_RE = re.compile(
    r"^(?:(?:Документ|Продукт|Заголовок|Раздел|Фрагмент):[^\n]*\n)+[ \t]*\n?",
    re.MULTILINE,
)


def clean_chunk_text(text):
    """Strip the `Документ:/Продукт:/Заголовок:/Раздел:/Фрагмент:` header block."""
    return _PREFIX_RE.sub("", text).strip()


def connect(db_path):
    """Open a SQLite connection with foreign keys enabled."""
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys=ON")
    return con


def load_chunks(db_path):
    """Return (chunks, by_id) with cleaned texts.

    Each chunk dict: {id, text, source, chunk_index}.
    """
    con = connect(db_path)
    try:
        rows = con.execute(
            "SELECT id, text, source, chunk_index FROM chunks ORDER BY id"
        ).fetchall()
    finally:
        con.close()
    chunks = [
        {
            "id": cid,
            "text": clean_chunk_text(text),
            "source": source,
            "chunk_index": chunk_index,
        }
        for cid, text, source, chunk_index in rows
    ]
    return chunks, {c["id"]: c for c in chunks}


def load_vectors(db_path):
    """Return (ids, matrix) with document vectors from the embeddings table.

    ids: list of chunk ids in the same order as matrix rows.
    matrix: np.ndarray of shape (n_chunks, dim), float32.
    """
    con = connect(db_path)
    try:
        rows = con.execute(
            "SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id"
        ).fetchall()
    finally:
        con.close()
    ids = [r[0] for r in rows]
    matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return ids, matrix
#Developed by YaAgent / SagaAI Platform, 2026. https://github.com/TVD2100/sagaai-platform
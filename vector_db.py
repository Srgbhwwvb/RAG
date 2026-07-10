from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

from embedding import SentenceEmbedder
from preprocessing import Preprocessor
from sentence_transformers import CrossEncoder


class VectorDB:
    """Vector database wrapper."""

    def __init__(
        self,
        path2data: str | Path,
        loaders: Mapping[str, Any],
        chunk_length: int,
        chunk_overlap: int,
        embedder: SentenceEmbedder | None = None,
        db_path: str | Path | None = None,
        collection_name: str = "Embeddings",
    ) -> None:
        self.path2data = Path(path2data)
        self.loaders = dict(loaders)
        self.chunk_length = chunk_length
        self.chunk_overlap = chunk_overlap
        self.db_path = Path(db_path) if db_path else self.path2data.parent / "db"
        self.collection_name = collection_name
        self.sentence_embedder = embedder or SentenceEmbedder()
        self.cross_encoder = None
        self.preprocessor = Preprocessor(
            self.path2data,
            self.loaders,
            self.chunk_length,
            self.chunk_overlap,
        )
        self._start_db()

    def _start_db(self) -> None:
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

    def _reset_collection(self) -> None:
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

    def preprocess_data(self) -> list[Document]:
        return self.preprocessor.preprocess_data()

    def get_embeddings(self, chunks: list[Document]) -> np.ndarray:
        return self.sentence_embedder.get_embeddings(chunks)

    def get_uuids(self, chunks: list[Document]) -> list[str]:
        """Generate deterministic ids from content and key metadata."""
        uuids = []
        for chunk in chunks:
            text = chunk.page_content
            source = chunk.metadata.get("source", "")
            page = chunk.metadata.get("page", -1)
            combined = f"{source}:{page}:{text}"
            hash_obj = sha256(combined.encode("utf-8"))
            uuids.append(hash_obj.hexdigest())
        return uuids

    def fill_db(self, *, reset_collection: bool = True) -> int:
        """Split, embed, deduplicate, and store the documents."""
        if reset_collection:
            self._reset_collection()

        chunks = self.preprocess_data()
        embeddings = self.get_embeddings(chunks)
        uuids = self.get_uuids(chunks)

        documents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        existing_ids = set(self.collection.get()["ids"])
        unique_ids = []
        unique_embeddings = []
        unique_docs = []
        unique_meta = []

        seen_ids = set()
        for i, uuid in enumerate(uuids):
            if uuid in existing_ids or uuid in seen_ids:
                continue
            seen_ids.add(uuid)
            unique_ids.append(uuid)
            unique_embeddings.append(embeddings[i].tolist())
            unique_docs.append(documents[i])
            unique_meta.append(metadatas[i])

        if unique_ids:
            self.collection.add(
                ids=unique_ids,
                embeddings=unique_embeddings,
                documents=unique_docs,
                metadatas=unique_meta,
            )

        return len(unique_ids)

    def remove_collection(self) -> None:
        self._reset_collection()

    def _init_cross_encoder(self, cross_encoder_name: str) -> None:
        self.cross_encoder = CrossEncoder(cross_encoder_name)

    def rerank(
        self,
        prompt: str,
        documents: list[str],
        top_n: int,
        cross_encoder_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the highest-scoring documents using cross-encoder."""
        if self.cross_encoder is None:
            self._init_cross_encoder(cross_encoder_name)
        pairs = [(prompt, doc) for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        scores = np.array(scores)
        sorted_indices = np.argsort(scores)[::-1]
        top_indices = sorted_indices[:top_n]
        top_docs = np.array(documents)[top_indices]
        top_scores = scores[top_indices]
        return top_docs, top_scores

    def query(
        self,
        prompt: str,
        threshold: float = 0.3,
        top_n: int = 5,
        top_k: int = 3,
        use_rerank: bool = True,
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.collection.count() == 0:
            return None, None
    
        result = self.collection.get(include=["embeddings", "documents"])
        stored_embeddings = np.array(result["embeddings"])
        stored_docs = result["documents"]
    
        prompt_embedding = self.sentence_embedder.get_embeddings([Document(page_content=prompt)])[0]
        similarities = cosine_similarity([prompt_embedding], stored_embeddings)[0]
    
        mask = similarities > threshold
        if not np.any(mask):
            return None, None
    
        filtered_docs = np.array(stored_docs)[mask]
        filtered_scores = similarities[mask]
    
        sorted_indices = np.argsort(filtered_scores)[::-1]
        top_n = min(top_n, len(sorted_indices))
        candidate_indices = sorted_indices[:top_n]
        candidate_docs = filtered_docs[candidate_indices]
        candidate_scores = filtered_scores[candidate_indices]
    
        if use_rerank:
            docs, scores = self.rerank(prompt, candidate_docs.tolist(), top_k, cross_encoder_name)
            if len(scores) == 0 or np.max(scores) < 0.5:
                return None, None
            return docs, scores
        else:
            final_docs = candidate_docs[:top_k]
            final_scores = candidate_scores[:top_k]
            return final_docs, final_scores
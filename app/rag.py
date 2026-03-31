from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import (
    KNOWLEDGE_DIR,
    VECTOR_EMB_PATH,
    VECTOR_INDEX_PATH,
    VECTOR_META_PATH,
    settings,
)
from app.knowledge import city_variants, canonical_city_key, slugify_city

if settings.rag_use_faiss:
    try:
        import faiss
    except Exception:  # pragma: no cover
        faiss = None
else:
    faiss = None

if settings.rag_use_transformers:
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except Exception:  # pragma: no cover
        CrossEncoder = None
        SentenceTransformer = None
else:
    CrossEncoder = None
    SentenceTransformer = None


@dataclass
class RetrievedDoc:
    source: str
    text: str
    score: float


class EmbeddingBackend:
    def __init__(self, model_name: str, dim: int = 384) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = None
        if SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(model_name)
                self.dim = int(self._model.get_sentence_embedding_dimension())
            except Exception:
                self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vectors, dtype=np.float32)
        return self._hash_embeddings(texts)

    def _hash_embeddings(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower())
            if not tokens:
                continue
            for token in tokens:
                idx = hash(token) % self.dim
                mat[i, idx] += 1.0
            norm = float(np.linalg.norm(mat[i])) or 1.0
            mat[i] = mat[i] / norm
        return mat


class RerankerBackend:
    def __init__(self, model_name: str) -> None:
        self._model = None
        if CrossEncoder is not None:
            try:
                self._model = CrossEncoder(model_name)
            except Exception:
                self._model = None

    def rerank(self, query: str, candidates: list[RetrievedDoc], top_k: int) -> list[RetrievedDoc]:
        if not candidates:
            return []
        if self._model is None:
            return candidates[:top_k]

        pairs = [[query, c.text] for c in candidates]
        try:
            scores = self._model.predict(pairs)
        except Exception:
            return candidates[:top_k]

        scored = [
            RetrievedDoc(source=c.source, text=c.text, score=round(float(scores[i]), 4))
            for i, c in enumerate(candidates)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


class FaissVectorStore:
    def __init__(
        self,
        index_path: Path = VECTOR_INDEX_PATH,
        meta_path: Path = VECTOR_META_PATH,
        emb_path: Path = VECTOR_EMB_PATH,
        embedding_model: str = settings.rag_embedding_model,
        reranker_model: str = settings.rag_reranker_model,
    ) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.emb_path = emb_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.emb_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedder = EmbeddingBackend(model_name=embedding_model)
        self.reranker = RerankerBackend(model_name=reranker_model)
        self.dimension = self.embedder.dim

        self.metadata: list[dict[str, str]] = []
        self.index = None
        self.fallback_vectors: np.ndarray | None = None
        self._load()

    def _new_index(self):
        if faiss is None:
            raise RuntimeError("faiss is not available. Install faiss-cpu.")
        return faiss.IndexFlatIP(self.dimension)

    def _load(self) -> None:
        if self.meta_path.exists():
            self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self.index_path.exists() and faiss is not None:
            self.index = faiss.read_index(str(self.index_path))
        elif faiss is not None:
            self.index = self._new_index()
        if self.emb_path.exists():
            try:
                self.fallback_vectors = np.load(str(self.emb_path)).astype(np.float32)
            except Exception:
                self.fallback_vectors = None

    def clear(self) -> None:
        self.metadata = []
        if faiss is not None:
            self.index = self._new_index()
        self.fallback_vectors = None
        if self.emb_path.exists():
            self.emb_path.unlink()

    def add_documents(self, docs: list[dict[str, str]]) -> None:
        if not docs:
            self.clear()
            self.persist()
            return

        texts = [d["text"] for d in docs]
        vectors = self.embedder.encode(texts)
        self.fallback_vectors = vectors

        if faiss is not None:
            if vectors.shape[1] != self.dimension:
                self.dimension = int(vectors.shape[1])
                self.index = self._new_index()
            self.index.add(vectors)
        self.metadata = [{"source": d["source"], "text": d["text"]} for d in docs]
        self.persist()

    def persist(self) -> None:
        if faiss is None or self.index is None:
            if self.fallback_vectors is not None:
                np.save(str(self.emb_path), self.fallback_vectors)
            self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        faiss.write_index(self.index, str(self.index_path))
        if self.fallback_vectors is not None:
            np.save(str(self.emb_path), self.fallback_vectors)
        self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int = 20,
        city: str | None = None,
    ) -> list[RetrievedDoc]:
        if not query:
            return []
        expanded_query = self._expand_query(query)

        if faiss is None or self.index is None:
            if self.fallback_vectors is None or len(self.metadata) == 0:
                return []
            qv = self.embedder.encode([expanded_query])
            sims = (self.fallback_vectors @ qv[0]).tolist()
            idx_scores = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
            candidates = [
                RetrievedDoc(
                    source=self.metadata[idx]["source"],
                    text=self.metadata[idx]["text"],
                    score=round(float(score), 4),
                )
                for idx, score in idx_scores[: min(max(top_k, candidate_k), len(idx_scores))]
            ]
            candidates = self._filter_candidates_by_city(candidates, city=city)
            reranked = self.reranker.rerank(query=expanded_query, candidates=candidates, top_k=top_k)
            return self._keyword_boost(expanded_query, reranked, top_k, city=city)

        if self.index.ntotal == 0:
            return []

        qv = self.embedder.encode([expanded_query])
        cand_k = min(max(top_k, candidate_k), int(self.index.ntotal))
        scores, indices = self.index.search(qv, cand_k)

        candidates: list[RetrievedDoc] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            candidates.append(
                RetrievedDoc(source=item["source"], text=item["text"], score=round(float(score), 4))
            )

        candidates = self._filter_candidates_by_city(candidates, city=city)
        reranked = self.reranker.rerank(query=expanded_query, candidates=candidates, top_k=top_k)
        return self._keyword_boost(expanded_query, reranked, top_k, city=city)

    def _expand_query(self, query: str) -> str:
        expanded = query
        mappings = {
            "杭州": "Hangzhou",
            "西湖": "West Lake",
            "洛杉矶": "Los Angeles",
            "纽约": "New York",
            "上海": "Shanghai",
            "北京": "Beijing",
        }
        for zh, en in mappings.items():
            if zh in query and en.lower() not in expanded.lower():
                expanded += f" {en}"
        return expanded

    def _filter_candidates_by_city(self, docs: list[RetrievedDoc], city: str | None) -> list[RetrievedDoc]:
        if not city:
            return docs
        variants = city_variants(city)
        canonical = canonical_city_key(city)
        file_hints = {
            *[slugify_city(v) for v in variants if v],
            canonical.replace(" ", "_"),
        }
        if canonical == "hangzhou":
            file_hints.add("hz_travel_tips")
        if canonical == "los angeles":
            file_hints.add("la_travel_tips")

        filtered: list[RetrievedDoc] = []
        for doc in docs:
            source_lower = doc.source.lower()
            text_head = "\n".join(doc.text.splitlines()[:6]).lower()
            is_generic = "travel_planning_rules" in source_lower
            city_match = any(hint and hint in source_lower for hint in file_hints) or any(
                v.lower() in text_head for v in variants if len(v) > 2 or re.search(r"[\u4e00-\u9fff]", v)
            )
            if is_generic or city_match:
                filtered.append(doc)
        return filtered or [doc for doc in docs if "travel_planning_rules" in doc.source.lower()] or docs

    def _keyword_boost(
        self,
        query: str,
        docs: list[RetrievedDoc],
        top_k: int,
        city: str | None = None,
    ) -> list[RetrievedDoc]:
        if not docs:
            return docs
        q_tokens = set(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", query.lower()))
        city_key = canonical_city_key(city) if city else ""
        boosted: list[RetrievedDoc] = []
        for d in docs:
            source = d.source.lower()
            text = d.text.lower()
            token_hits = sum(1 for t in q_tokens if t in text or t in source)
            score = float(d.score) + 0.08 * token_hits

            # City/topic file bonus when query and source align.
            if ("hangzhou" in query.lower() or "杭州" in query) and "hz_travel_tips" in source:
                score += 0.6
            if ("west lake" in query.lower() or "西湖" in query) and "hz_travel_tips" in source:
                score += 0.4
            if ("los angeles" in query.lower() or "洛杉矶" in query) and "la_travel_tips" in source:
                score += 0.5
            if city_key == "los angeles" and ("los_angeles" in source or "la_travel_tips" in source):
                score += 0.8
            if city_key == "hangzhou" and ("hangzhou" in source or "hz_travel_tips" in source):
                score += 0.8
            if city_key and city_key.replace(" ", "_") in source:
                score += 0.8

            boosted.append(RetrievedDoc(source=d.source, text=d.text, score=round(score, 4)))

        boosted.sort(key=lambda x: x.score, reverse=True)
        return boosted[:top_k]


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def load_knowledge_docs(
    path: Path = KNOWLEDGE_DIR,
    chunk_size: int = settings.rag_chunk_size,
    overlap: int = settings.rag_chunk_overlap,
) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    if not path.exists():
        return docs
    for file_path in path.glob("*.md"):
        content = file_path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(chunk_text(content, chunk_size=chunk_size, overlap=overlap), 1):
            docs.append({"source": f"{file_path.name}#chunk{idx}", "text": chunk})
    return docs


def build_vector_store(force: bool = False) -> FaissVectorStore:
    store = FaissVectorStore()
    has_data = bool(store.metadata)
    docs = load_knowledge_docs()
    latest_sources = {d["source"] for d in docs}
    indexed_sources = {m["source"] for m in store.metadata}
    source_mismatch = latest_sources != indexed_sources
    broken_faiss_state = faiss is not None and (
        store.index is None or store.index.ntotal != len(store.metadata)
    )
    broken_fallback_state = faiss is None and (
        store.fallback_vectors is None or len(store.metadata) != len(store.fallback_vectors)
    )

    if force or not has_data or source_mismatch or broken_faiss_state or broken_fallback_state:
        store.clear()
        store.add_documents(docs)
    return store

from __future__ import annotations

from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token.strip()]


def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score - min_score < 1e-12:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)


class HybridRetriever:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks

    def _filter_indices(self, subject: str | None = None, chapter: str | None = None) -> list[int]:
        indices: list[int] = []
        chapter_filter = (chapter or "").strip().lower()

        for index, chunk in enumerate(self.chunks):
            subject_ok = True
            chapter_ok = True

            if subject:
                subject_ok = str(chunk.get("subject", "")).strip().lower() == subject.strip().lower()

            if chapter_filter:
                chapter_ok = chapter_filter in str(chunk.get("chapter", "")).strip().lower()

            if subject_ok and chapter_ok:
                indices.append(index)

        return indices

    def retrieve(
        self,
        query: str,
        subject: str | None = None,
        chapter: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not self.chunks:
            return []

        candidate_indices = self._filter_indices(subject=subject, chapter=chapter)
        if not candidate_indices:
            return []

        candidate_chunks = [self.chunks[i] for i in candidate_indices]
        corpus = [str(chunk.get("text", "")) for chunk in candidate_chunks]
        tokenized_corpus = [_tokenize(text) for text in corpus]
        query_tokens = _tokenize(query) or ["tax"]

        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = np.array(bm25.get_scores(query_tokens), dtype=float)

        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(corpus)
        query_vector = tfidf.transform([query])
        cosine_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

        combined = 0.5 * _normalize(bm25_scores) + 0.5 * _normalize(cosine_scores)
        ranked_positions = np.argsort(combined)[::-1][: max(1, top_k)]

        results: list[dict[str, Any]] = []
        for position in ranked_positions:
            chunk = dict(candidate_chunks[int(position)])
            chunk["retrieval_score"] = float(combined[int(position)])
            results.append(chunk)

        return results

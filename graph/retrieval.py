import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pinecone import Pinecone
from openai import AsyncOpenAI, OpenAI

from .models import RetrievalResult
from utils.logger import logger
from opik import track, opik_context

# OpenAI Client Setup
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


dense_immi_vector_store = pc.Index(host=os.getenv("PINECONE_IMMI_DENSE_INDEX"))
sparse_immi_vector_store = pc.Index(host=os.getenv("PINECONE_IMMI_SPARSE_INDEX"))

# Precompile regex patterns for performance
_whitespace_pattern = re.compile(r"\s+")
_xml_tag_pattern = re.compile(r"<[^>]+>")
_special_section_pattern = re.compile(
    r"<(?:questions|tags)>.*?</(?:questions|tags)>", re.IGNORECASE | re.DOTALL
)

# Global embeddings cache for example selector
EXAMPLE_EMBEDDINGS: List[List[float]] = []
QUERY_EMBEDDINGS_CACHE: Dict[str, List[float]] = {}


def preprocess_text(text: str) -> str:
    """Preprocess text by removing XML sections and normalizing whitespace"""
    text = _special_section_pattern.sub("", text)
    text = _xml_tag_pattern.sub("", text)
    return _whitespace_pattern.sub(" ", text).strip()


class VectorStore:
    """Vector store implementation using Pinecone"""

    def __init__(self, index_name: str):
        self.index_name = index_name
        self._index = None

    @property
    def index(self):
        if self._index is None:
            self._index = pc.Index(host=self.index_name)
        return self._index

    async def similarity_search(
        self, query_embedding: List[float], k: int = 3
    ) -> List[tuple[Dict, float]]:
        try:
            # Query Pinecone directly with the provided embedding
            # No need to generate embeddings again
            query_response = self.index.query(
                vector=query_embedding, top_k=k, include_metadata=True
            )

            # Format results
            results = []
            for match in query_response.matches:
                doc = {
                    "page_content": match.metadata.get("text", ""),
                    "metadata": match.metadata,
                }
                results.append((doc, match.score))

            return results

        except Exception as e:
            logger.error("Vector store query failed", extra={"error": str(e)})
            raise


# TODO: parameters can be removed once dynamic few shot is totally descoped.
@dataclass
class RetrieveTool:
    name: str = "RETRIEVE"
    description: str = "Retrieve information related to a query"
    parameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to retrieve information for",
                }
            },
            "required": ["query"],
        }
    )

    @track(capture_input=False)
    async def execute(self, query: str, language: str) -> List[RetrievalResult]:
        try:
            logger.info(
                "Starting retrieval",
                extra={"action": "retrieval_start", "query": query},
            )

            # Get embeddings using OpenAI directly
            query_embedding = await self._get_embedding(query)

            # Get raw results from vector store
            doc_score_pairs = await vector_store.similarity_search(query_embedding)

            # Process results with threshold
            if doc_score_pairs:
                best_score = doc_score_pairs[0][1]
                threshold = max(0.7, best_score * 0.9)

                # Process and filter results in a single pass
                processed_results = []

                for doc, score in doc_score_pairs:
                    if score >= threshold:
                        processed_content = preprocess_text(doc["page_content"])

                        processed_results.append(
                            RetrievalResult(
                                content=processed_content,
                                score=score,
                                metadata=doc["metadata"],
                            )
                        )

                opik_context.update_current_span(
                    name="retrieval",
                    input={"query": query},
                    output={"docs": processed_results},
                )

                return processed_results

            return []

        except Exception as e:
            logger.error("Retrieval failed", extra={"error": str(e)})
            raise

    async def _get_embedding(self, text: str) -> List[float]:
        response = await client.embeddings.create(
            model="text-embedding-ada-002", input=text
        )
        return response.data[0].embedding


# Initialize single vector store instance
vector_store = VectorStore(os.getenv("PINECONE_RESUME_INDEX"))


@dataclass
class HybridRetrieveTool:
    """
    Hybrid retrieval tool that:
      1) Searches both dense and sparse Pinecone indexes using .search()
      2) Merges and deduplicates results by _id
      3) Applies a language filter (default 'en'; switches to 'es' if query appears Spanish)
      4) Reranks merged results using Pinecone inference.rerank
      5) Gracefully falls back to top-N merged results on 429 or rerank failures
      6) Returns RetrievalResult instances compatible with the existing flow
    """

    name: str = "HYBRID_RETRIEVE"
    description: str = "Hybrid dense+sparse retrieval with reranking"
    parameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to retrieve information for",
                }
            },
            "required": ["query"],
        }
    )

    _top_k: int = field(default_factory=lambda: int(os.getenv("HYBRID_TOP_K", "6")))
    _top_n: int = field(
        default_factory=lambda: int(os.getenv("HYBRID_RERANK_TOP_N", "3"))
    )
    _namespace: Optional[str] = field(
        default_factory=lambda: os.getenv("PINECONE_IMMI_NAMESPACE") or None
    )

    @staticmethod
    def _extract_hits(result: Any) -> List[Dict[str, Any]]:
        """
        Normalize Pinecone search() response into a list of hit dicts.
        Expected shape:
          { 'result': { 'hits': [ {'_id': '...', '_score': float, 'fields': {...}}, ... ] } }
        """
        try:
            root = result
            if hasattr(root, "result"):
                root = getattr(root, "result")
            elif isinstance(root, dict):
                root = root.get("result", {})

            if hasattr(root, "hits"):
                hits = getattr(root, "hits") or []
                return list(hits)
            if isinstance(root, dict):
                return list(root.get("hits", []) or [])
            return []
        except Exception as e:
            logger.warning(
                "Failed to parse Pinecone search() response", extra={"error": str(e)}
            )
            return []

    @staticmethod
    def _merge_and_dedupe_hits(
        dense_hits: List[Dict[str, Any]],
        sparse_hits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge hits from dense and sparse searches by _id into a normalized list.
        - Deduplicates by _id, keeping the higher _score as representative
        - Preserves 'fields' and per-source scores in 'score_by_source'
        - Adds a top-level 'chunk_text' field for reranking (rank_fields=["chunk_text"]) with sensible fallbacks

        Returns documents suitable for Pinecone rerank and downstream mapping, e.g.:
          {
            "_id": "...",
            "_score": 0.42,
            "chunk_text": "...",
            "fields": {...},
            "score_by_source": {"dense": 0.40, "sparse": 0.38},
            "index_source": "dense"|"sparse"|"merged"
          }
        """
        by_id: Dict[str, Dict[str, Any]] = {}

        def _extract_chunk_text(fields: Dict[str, Any]) -> str:
            # Provide a robust fallback chain so reranker always sees 'chunk_text'
            return (
                (fields or {}).get("chunk_text")
                or (fields or {}).get("text")
                or (fields or {}).get("content")
                or ""
            )

        def _absorb(hit: Dict[str, Any], source: str):
            _id = hit.get("_id")
            if not _id:
                return
            score = hit.get("_score", 0.0)
            fields = hit.get("fields") or {}
            entry = by_id.get(_id)
            if entry is None:
                by_id[_id] = {
                    "_id": _id,
                    "_score": score,
                    "fields": fields,
                    "chunk_text": _extract_chunk_text(fields),
                    "score_by_source": {source: score},
                    "index_source": source,
                }
            else:
                entry["score_by_source"][source] = score
                # Prefer higher score for representative _score and refresh text/fields if richer
                if score > entry["_score"]:
                    entry["_score"] = score
                    if fields:
                        entry["fields"] = fields
                        entry["chunk_text"] = _extract_chunk_text(fields)
                    entry["index_source"] = "merged"

        for h in dense_hits:
            _absorb(h, "dense")
        for h in sparse_hits:
            _absorb(h, "sparse")

        merged = list(by_id.values())
        merged.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
        return merged

    @staticmethod
    def _to_retrieval_result(
        hit: Dict[str, Any], *, namespace: Optional[str], rerank_score: Optional[float]
    ) -> RetrievalResult:
        """Convert a merged 'hit' dict into our RetrievalResult model.

        Prefer top-level 'chunk_text' (added during merge) and fall back to common field names.
        """
        fields = hit.get("fields") or {}
        text = (
            hit.get("chunk_text")
            or fields.get("chunk_text")
            or fields.get("text")
            or ""
        )
        content = preprocess_text(text)
        original_score = hit.get("_score", 0.0)
        metadata: Dict[str, Any] = {
            "id": hit.get("_id"),
            "namespace": namespace,
            "source": fields.get("source"),
            "source_document": fields.get("source_document"),
            "section_title": fields.get("section_title"),
            "parent_section": fields.get("parent_section"),
            "language": fields.get("language"),
            "keywords": fields.get("keywords"),
            "original_score": original_score,
            "rerank_score": rerank_score,
            "score_by_source": hit.get("score_by_source", {}),
            "index_source": hit.get("index_source", "merged"),
        }
        final_score = float(
            rerank_score if rerank_score is not None else original_score or 0.0
        )
        return RetrievalResult(content=content, score=final_score, metadata=metadata)

    @track(capture_input=False)
    async def execute(self, query: str, language: str) -> List[RetrievalResult]:
        """
        Execute hybrid retrieval for the 'query'. Steps:
          - dense.search + sparse.search
          - merge/dedupe
          - language filter (default 'en', or 'es' if query seems Spanish)
          - rerank with Pinecone inference; fallback to top-N by merged score on 429
        """
        try:
            logger.info(
                "Starting hybrid retrieval",
                extra={
                    "action": "hybrid_retrieval_start",
                    "query": query,
                    "namespace": self._namespace,
                    "top_k": self._top_k,
                    "top_n": self._top_n,
                },
            )

            if language == "spanish":
                language = "es"
            else:
                language = "en"

            # 1) Run dense and sparse searches
            dense_results = dense_immi_vector_store.search(
                namespace=self._namespace,
                query={
                    "top_k": self._top_k,
                    "inputs": {"text": query},
                    "filter": {"language": language},
                },
            )
            sparse_results = sparse_immi_vector_store.search(
                namespace=self._namespace,
                query={
                    "top_k": self._top_k,
                    "inputs": {"text": query},
                    "filter": {"language": language},
                },
            )

            dense_hits = self._extract_hits(dense_results)
            sparse_hits = self._extract_hits(sparse_results)
            logger.debug(
                "Raw hits",
                extra={
                    "dense_count": len(dense_hits),
                    "sparse_count": len(sparse_hits),
                },
            )

            # 2) Merge + dedupe
            merged_hits = self._merge_and_dedupe_hits(dense_hits, sparse_hits)
            logger.info(
                "Hybrid Retrieval success: Overall documents count",
                extra={"merged_count": len(merged_hits)},
            )

            # 5) Rerank with graceful fallback on 429
            reranked_items: List[RetrievalResult] = []
            try:
                result = pc.inference.rerank(
                    model="pinecone-rerank-v0",
                    query=query,
                    documents=merged_hits,
                    rank_fields=["chunk_text"],
                    top_n=self._top_n,
                    return_documents=True,
                    parameters={"truncate": "END"},
                )

                data_list = getattr(result, "data", None)
                if data_list is None and isinstance(result, dict):
                    data_list = result.get("data", [])

                id_to_hit = {h.get("_id"): h for h in merged_hits}

                for item in data_list or []:
                    document = getattr(item, "document", None)
                    if document is None and isinstance(item, dict):
                        document = item.get("document", {})
                    score = getattr(item, "score", None)
                    if score is None and isinstance(item, dict):
                        score = item.get("score")

                    doc_id = (
                        getattr(document, "_id", None) if document is not None else None
                    )
                    if doc_id is None and isinstance(document, dict):
                        doc_id = document.get("_id")

                    if not doc_id:
                        continue

                    base_hit = id_to_hit.get(doc_id)
                    if not base_hit:
                        continue

                    reranked_items.append(
                        self._to_retrieval_result(
                            base_hit, namespace=self._namespace, rerank_score=score
                        )
                    )

                if not reranked_items:
                    logger.info("Rerank returned empty; falling back to merged top-N")
                    reranked_items = [
                        self._to_retrieval_result(
                            h, namespace=self._namespace, rerank_score=None
                        )
                        for h in merged_hits[: self._top_n]
                    ]

            except Exception as e:
                msg = str(e)
                is_429 = (
                    "429" in msg
                    or getattr(e, "status", None) == 429
                    or getattr(e, "status_code", None) == 429
                )
                if is_429:
                    logger.warning(
                        "Rerank rate-limited (429). Falling back to merged top-N.",
                        extra={"error": msg},
                    )
                else:
                    logger.error(
                        "Rerank failed; falling back to merged top-N.",
                        extra={"error": msg},
                    )
                reranked_items = [
                    self._to_retrieval_result(
                        h, namespace=self._namespace, rerank_score=None
                    )
                    for h in merged_hits[: self._top_n]
                ]

            # Filter out docs with rerank score below a threshold
            RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.0004"))
            reranked_items = [
                r
                for r in reranked_items
                if (
                    r.metadata.get("rerank_score") is None  # keep fallback docs
                    or r.metadata["rerank_score"] > RERANK_MIN_SCORE
                )
            ]

            logger.info(
                "Hybrid Retrieval success: Reranked documents",
                extra={"reranked_items count": len(reranked_items)},
            )

            opik_context.update_current_span(
                name="hybrid_retrieval",
                input={"query": query},
                output={"docs": [r.to_dict() for r in reranked_items]},
            )
            return reranked_items

        except Exception as e:
            logger.error("Hybrid retrieval failed", extra={"error": str(e)})
            raise

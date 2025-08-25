import os
import re
from typing import List, Dict, Any
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
            self._index = pc.Index(self.index_name)
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
    async def execute(self, query: str) -> List[RetrievalResult]:
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
vector_store = VectorStore(index_name="langchain-chatraghu-embeddings")

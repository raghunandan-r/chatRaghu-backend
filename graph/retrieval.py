import os
import re
import numpy as np
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

from pinecone import Pinecone
from openai import AsyncOpenAI, OpenAI

from .models import RetrievalResult, Tool
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


class VectorStore(BaseModel):
    """Vector store implementation using Pinecone"""

    index_name: str
    index: Optional[
        Any
    ] = None  # Change Index to Any since Pinecone's type isn't Pydantic compatible

    def __init__(self, **data):
        super().__init__(**data)
        self.index = pc.Index(self.index_name)

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


class RetrieveTool(Tool):
    name: str = "RETRIEVE"
    description: str = "Retrieve information related to a query"
    parameters: Dict[str, Any] = Field(
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
    async def execute(self, query: str) -> tuple[str, List[RetrievalResult]]:
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
                serialized_chunks = []

                for doc, score in doc_score_pairs:
                    if score >= threshold:
                        processed_content = preprocess_text(doc["page_content"])
                        serialized_chunks.append(
                            f"Content: {processed_content} (Score: {score:.2f})"
                        )
                        processed_results.append(
                            RetrievalResult(
                                content=processed_content,
                                score=score,
                                metadata=doc["metadata"],
                            )
                        )

                opik_context.update_current_span(
                    name="chunk_retrieval",
                    input={"query": query},
                    output={"docs": processed_results},
                )

                return "\n\n".join(serialized_chunks), processed_results

            return "", []

        except Exception as e:
            logger.error("Retrieval failed", extra={"error": str(e)})
            raise

    async def _get_embedding(self, text: str) -> List[float]:
        response = await client.embeddings.create(
            model="text-embedding-ada-002", input=text
        )
        return response.data[0].embedding


class ExampleSelector(BaseModel):
    examples: List[Dict[str, str]]

    @classmethod
    async def initialize_examples(cls, examples: List[Dict[str, str]]):
        """Initialize global example embeddings at server startup"""
        global EXAMPLE_EMBEDDINGS
        if not EXAMPLE_EMBEDDINGS and examples:
            embeddings_response = await client.embeddings.create(
                model="text-embedding-ada-002",
                input=[ex.get("user_query", ex.get("query", "")) for ex in examples],
            )
            EXAMPLE_EMBEDDINGS = [data.embedding for data in embeddings_response.data]

    async def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query using global cache"""
        global QUERY_EMBEDDINGS_CACHE
        if query in QUERY_EMBEDDINGS_CACHE:
            return QUERY_EMBEDDINGS_CACHE[query]

        response = await client.embeddings.create(
            model="text-embedding-ada-002", input=[query]
        )
        embedding = response.data[0].embedding
        QUERY_EMBEDDINGS_CACHE[query] = embedding
        return embedding

    async def get_relevant_examples(
        self, query: str, k: int = 3
    ) -> List[Dict[str, str]]:
        """Get the most relevant examples using global embeddings"""
        # If no examples are available, return empty list
        if not self.examples or not EXAMPLE_EMBEDDINGS:
            logger.warning(
                "No examples available for few-shot selection",
                extra={
                    "examples_count": len(self.examples),
                    "embeddings_count": len(EXAMPLE_EMBEDDINGS)
                    if EXAMPLE_EMBEDDINGS
                    else 0,
                },
            )
            return []

        query_embedding = await self.get_query_embedding(query)

        # Calculate similarities
        similarities = [
            np.dot(query_embedding, ex_embedding)
            / (np.linalg.norm(query_embedding) * np.linalg.norm(ex_embedding))
            for ex_embedding in EXAMPLE_EMBEDDINGS
        ]

        # Get top k examples, but ensure we don't exceed available examples
        k = min(k, len(self.examples), len(EXAMPLE_EMBEDDINGS))
        if k == 0:
            return []

        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.examples[i] for i in top_indices]


# Initialize single vector store instance
vector_store = VectorStore(index_name="langchain-chatraghu-embeddings")

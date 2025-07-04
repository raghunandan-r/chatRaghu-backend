"""
Evaluation Client for ChatRaghu Backend

This module provides a client interface to communicate with the separate
evaluation service for processing conversation evaluations.
"""

import os
import httpx
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel
from utils.logger import logger
from evaluation_models import ConversationFlow


class EvaluationRequest(BaseModel):
    """Request model for evaluation service"""

    thread_id: str
    query: str
    response: str
    retrieved_docs: Optional[List[Dict[str, str]]] = None
    conversation_flow: Union[ConversationFlow, Dict[str, Any]]


class EvaluationResponse(BaseModel):
    """Response model from evaluation service"""

    thread_id: str
    success: bool
    evaluation_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime


class EvaluationClient:
    """Client for communicating with the evaluation service"""

    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 30):
        """
        Initialize the evaluation client.

        Args:
            base_url: Base URL of the evaluation service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._client

    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the evaluation service is healthy.

        Returns:
            Health status information
        """
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Health check failed", extra={"error": str(e)})
            raise

    async def evaluate_conversation_async(
        self,
        thread_id: str,
        query: str,
        response: str,
        conversation_flow: Union[ConversationFlow, Dict[str, Any]],
        retrieved_docs: Optional[List[Dict[str, str]]] = None,
    ) -> EvaluationResponse:
        """
        Submit a conversation for asynchronous evaluation.

        Args:
            thread_id: Unique identifier for the conversation
            query: Original user query
            response: Generated response
            conversation_flow: Complete conversation flow data (ConversationFlow object or dict)
            retrieved_docs: Retrieved documents (optional)

        Returns:
            Evaluation response indicating success/failure
        """
        try:
            # Handle both ConversationFlow objects and dictionaries
            if isinstance(conversation_flow, dict):
                node_count = len(conversation_flow.get("node_executions", []))
            else:
                node_count = len(conversation_flow.node_executions)

            request_data = EvaluationRequest(
                thread_id=thread_id,
                query=query,
                response=response,
                retrieved_docs=retrieved_docs,
                conversation_flow=conversation_flow,
            )

            # Log the exact data being sent for debugging
            request_json = request_data.model_dump(mode="json")
            logger.info(
                "Sending evaluation request data",
                extra={
                    "thread_id": thread_id,
                    "request_data": request_json,
                    "node_count": node_count,
                },
            )

            client = await self._get_client()
            response = await client.post(f"{self.base_url}/evaluate", json=request_json)
            response.raise_for_status()

            result = EvaluationResponse(**response.json())

            logger.info(
                "Successfully submitted evaluation request",
                extra={"thread_id": thread_id, "success": result.success},
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to submit evaluation request",
                extra={"thread_id": thread_id, "error": str(e)},
            )
            raise

    async def evaluate_conversation_sync(
        self,
        thread_id: str,
        query: str,
        response: str,
        conversation_flow: Union[ConversationFlow, Dict[str, Any]],
        retrieved_docs: Optional[List[Dict[str, str]]] = None,
    ) -> EvaluationResponse:
        """
        Submit a conversation for synchronous evaluation.

        Args:
            thread_id: Unique identifier for the conversation
            query: Original user query
            response: Generated response
            conversation_flow: Complete conversation flow data (ConversationFlow object or dict)
            retrieved_docs: Retrieved documents (optional)

        Returns:
            Evaluation response with results
        """
        try:
            # Handle both ConversationFlow objects and dictionaries
            if isinstance(conversation_flow, dict):
                node_count = len(conversation_flow.get("node_executions", []))
            else:
                node_count = len(conversation_flow.node_executions)

            request_data = EvaluationRequest(
                thread_id=thread_id,
                query=query,
                response=response,
                retrieved_docs=retrieved_docs,
                conversation_flow=conversation_flow,
            )

            # Log the exact data being sent for debugging
            request_json = request_data.model_dump(mode="json")
            logger.info(
                "Sending synchronous evaluation request data",
                extra={
                    "thread_id": thread_id,
                    "request_data": request_json,
                    "node_count": node_count,
                },
            )

            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/evaluate/sync", json=request_json
            )
            response.raise_for_status()

            result = EvaluationResponse(**response.json())

            logger.info(
                "Successfully completed synchronous evaluation",
                extra={"thread_id": thread_id, "success": result.success},
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to complete synchronous evaluation",
                extra={"thread_id": thread_id, "error": str(e)},
            )
            raise

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get evaluation service metrics.

        Returns:
            Service metrics information
        """
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/metrics")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get metrics", extra={"error": str(e)})
            raise


class MockEvaluationClient:
    async def evaluate_conversation_async(self, *args, **kwargs):
        # Return a dummy response
        return {
            "thread_id": kwargs.get("thread_id", "mock"),
            "success": True,
            "evaluation_result": None,
            "error": None,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def close(self):
        pass


# Global evaluation client instance
evaluation_client: Optional[EvaluationClient] = None


async def get_evaluation_client() -> EvaluationClient:
    """Get the global evaluation client instance"""
    global evaluation_client
    if evaluation_client is None:
        if os.getenv("MOCK_EVAL_CLIENT", "false").lower() == "true":
            evaluation_client = MockEvaluationClient()
        else:
            # Get configuration from environment
            eval_service_url = os.getenv("EVALUATION_SERVICE_URL")
            timeout = int(os.getenv("EVALUATION_SERVICE_TIMEOUT", "30"))
            evaluation_client = EvaluationClient(
                base_url=eval_service_url, timeout=timeout
            )
    return evaluation_client


async def close_evaluation_client():
    """Close the global evaluation client"""
    global evaluation_client
    if evaluation_client:
        await evaluation_client.close()
        evaluation_client = None

import pytest
from fastapi.testclient import TestClient
from app import app
from unittest.mock import patch


@pytest.mark.asyncio
@pytest.mark.skip(reason="Graph error handling needs to be implemented")
async def test_graph_timeout():
    """Test that graph timeout errors are handled properly"""
    client = TestClient(app)

    with patch(
        "graph.graph.StreamingStateGraph.execute_stream", side_effect=TimeoutError
    ):
        response = client.post(
            "/api/chat",
            headers={"X-API-Key": "test_api_key_123"},
            json={"messages": [{"role": "user", "content": "Test timeout"}]},
        )
        # Should handle timeout gracefully
        assert response.status_code in [500, 504]


@pytest.mark.asyncio
@pytest.mark.skip(reason="Graph error handling needs to be implemented")
async def test_graph_execution_error():
    """Test that graph execution errors are handled properly"""
    client = TestClient(app)

    with patch(
        "graph.graph.StreamingStateGraph.execute_stream",
        side_effect=Exception("Test error"),
    ):
        response = client.post(
            "/api/chat",
            headers={"X-API-Key": "test_api_key_123"},
            json={"messages": [{"role": "user", "content": "Test error"}]},
        )
        # Should handle errors gracefully
        assert response.status_code in [500, 504]

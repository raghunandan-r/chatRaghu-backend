import pytest
from fastapi import HTTPException
from app import safe_graph_execution, graph
from langchain_core.messages import HumanMessage
from unittest.mock import patch

@pytest.mark.asyncio
async def test_graph_timeout():
    messages = {"messages": [HumanMessage(content="Test timeout")]}
    config = {"configurable": {"thread_id": "test_thread"}}
    
    with patch("app.graph.astream", side_effect=TimeoutError):
        with pytest.raises(HTTPException) as exc_info:
            async for _ in safe_graph_execution(messages, "messages", config):
                pass
        assert exc_info.value.status_code == 504

@pytest.mark.asyncio
async def test_graph_execution_error():
    messages = {"messages": [HumanMessage(content="Test error")]}
    config = {"configurable": {"thread_id": "test_thread"}}
    
    with patch("app.graph.astream", side_effect=Exception("Test error")):
        with pytest.raises(HTTPException) as exc_info:
            async for _ in safe_graph_execution(messages, "messages", config):
                pass
        assert exc_info.value.status_code == 500 
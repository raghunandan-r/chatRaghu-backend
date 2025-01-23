import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app import (
    relevance_check,
    relevance_condition,
    query_or_respond,
    generate_with_retrieved_context,
    generate_with_persona,
    few_shot_selector
)

# Fixtures for common test data
@pytest.fixture
def messages_state():
    return {
        "messages": [
            HumanMessage(content="What are your technical skills?")
        ]
    }

@pytest.fixture
def messages_state_with_history():
    return {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Greetings"),
            HumanMessage(content="What are your technical skills?")
        ]
    }

# Test relevance_check node
@pytest.mark.asyncio
async def test_relevance_check(messages_state):
    with patch("app.llm.invoke") as mock_llm:
        mock_llm.return_value = AIMessage(content="CONTEXTUAL")
        result = relevance_check(messages_state)
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "CONTEXTUAL" in result["messages"][0].content

# Test relevance_condition
def test_relevance_condition_contextual():
    state = {
        "messages": [
            HumanMessage(content="What are your skills?"),
            AIMessage(content="CONTEXTUAL")
        ]
    }
    result = relevance_condition(state)
    assert result == "CONTEXTUAL"

def test_relevance_condition_irrelevant():
    state = {
        "messages": [
            HumanMessage(content="What's the weather like?"),
            AIMessage(content="IRRELEVANT")
        ]
    }
    result = relevance_condition(state)
    assert result == "IRRELEVANT"

# Test query_or_respond node
@pytest.mark.asyncio
async def test_query_or_respond(messages_state):
    with patch("app.llm.bind_tools") as mock_bind_tools:
        mock_llm_with_tools = AsyncMock()
        mock_llm_with_tools.invoke.return_value = AIMessage(
            content="Using retrieve tool to fetch information."
        )
        mock_bind_tools.return_value = mock_llm_with_tools
        
        result = query_or_respond(messages_state)
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 1

# Test generate_with_retrieved_context
@pytest.mark.asyncio
async def test_generate_with_retrieved_context():
    state = {
        "messages": [
            HumanMessage(content="What are your technical skills?"),
            AIMessage(content="Let me check that for you."),
            SystemMessage(content="Retrieved content about Python, FastAPI, and React skills")
        ]
    }
    
    with patch("app.llm.ainvoke") as mock_llm:
        mock_llm.return_value = AIMessage(
            content="Based on the retrieved information, the technical skills include Python, FastAPI, and React."
        )
        
        result = await generate_with_retrieved_context(state)
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 1

# Test generate_with_persona
@pytest.mark.asyncio
async def test_generate_with_persona(messages_state_with_history):
    with patch("app.llm.invoke") as mock_llm:
        mock_llm.return_value = AIMessage(
            content="Raghu possesses mastery over Python, FastAPI, and React."
        )
        
        result = await generate_with_persona(messages_state_with_history)
        assert isinstance(result, dict)
        assert "messages" in result
        assert "Raghu" in result["messages"][0].content

# Test complete graph flow
@pytest.mark.asyncio
async def test_complete_graph_flow():
    from app import graph
    
    messages = {
        "messages": [HumanMessage(content="What are your technical skills?")]
    }
    config = {"configurable": {"thread_id": "test_thread"}}
    
    async for msg, metadata in graph.astream(
        messages,
        stream_mode="messages",
        config=config
    ):
        # Check each node's execution
        assert metadata["langgraph_node"] in {
            "relevance_check",
            "query_or_respond",
            "tools",
            "generate_with_retrieved_context",
            "generate_with_persona"
        }
        
        # Verify message structure
        assert msg.content is not None
        if metadata["langgraph_node"] == "generate_with_persona":
            assert "Raghu" in msg.content 
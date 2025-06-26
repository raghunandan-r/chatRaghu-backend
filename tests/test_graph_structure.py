"""
Tests for the new graph structure organization.

This module tests that the refactored graph package structure works correctly
and maintains backward compatibility with existing imports.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch
from graph.models import (
    MessagesState,
    HumanMessage,
    AIMessage,
    ToolMessage,
    RetrievalResult,
)
from graph.nodes import RelevanceCheckNode, QueryOrRespondNode

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.graph_structure
class TestGraphStructure:
    """Test the new graph package structure."""

    def test_file_structure(self):
        """Test that all expected files exist in the new structure."""
        expected_files = [
            "graph/__init__.py",
            "graph/models.py",
            "graph/infrastructure.py",
            "graph/retrieval.py",
            "graph/nodes.py",
            "graph/prompt_templates.json",
        ]

        for file_path in expected_files:
            assert os.path.exists(
                file_path
            ), f"Expected file {file_path} does not exist"

    def test_models_import(self):
        """Test that models can be imported correctly."""
        from graph.models import (
            HumanMessage,
        )

        # Test that classes can be instantiated
        message = HumanMessage(content="Test message")
        assert message.content == "Test message"
        assert message.type == "human"

    def test_infrastructure_import(self):
        """Test that infrastructure classes can be imported correctly."""
        from graph.infrastructure import (
            Node,
            StreamingNode,
            StateGraph,
        )

        # Test that abstract base classes can be imported
        assert Node is not None
        assert StreamingNode is not None
        assert StateGraph is not None

    def test_retrieval_import(self):
        """Test that retrieval classes can be imported correctly."""
        from graph.retrieval import (
            VectorStore,
            RetrieveTool,
            ExampleSelector,
        )

        # Test that classes can be imported
        assert VectorStore is not None
        assert RetrieveTool is not None
        assert ExampleSelector is not None

    def test_nodes_import(self):
        """Test that node classes can be imported correctly."""
        from graph.nodes import (
            RelevanceCheckNode,
            QueryOrRespondNode,
            FewShotSelectorNode,
        )

        # Test that node classes can be imported
        assert RelevanceCheckNode is not None
        assert QueryOrRespondNode is not None
        assert FewShotSelectorNode is not None

    def test_main_package_import(self):
        """Test that the main package import works correctly."""
        from graph import (
            streaming_graph,
            MessagesState,
            HumanMessage,
        )

        # Test that main components can be imported
        assert streaming_graph is not None
        assert MessagesState is not None
        assert HumanMessage is not None

    def test_backward_compatibility(self):
        """Test that the new structure maintains backward compatibility."""
        # Test that the old import pattern still works
        from graph import MessagesState, HumanMessage

        # Test that we can create objects
        message = HumanMessage(content="Backward compatibility test")
        state = MessagesState(messages=[message], thread_id="test-123")

        assert len(state.messages) == 1
        assert state.thread_id == "test-123"
        assert state.messages[0].content == "Backward compatibility test"

    def test_message_state_functionality(self):
        """Test that MessagesState works correctly."""
        from graph.models import MessagesState, HumanMessage, AIMessage

        # Create a simple conversation
        human_msg = HumanMessage(content="Hello")
        ai_msg = AIMessage(content="Hi there!")

        state = MessagesState(messages=[human_msg, ai_msg], thread_id="test-thread")

        assert len(state.messages) == 2
        assert state.thread_id == "test-thread"
        assert state.messages[0].content == "Hello"
        assert state.messages[1].content == "Hi there!"

    def test_thread_store_functionality(self):
        """Test that thread store functionality works."""
        from graph.models import MessagesState, HumanMessage, THREAD_MESSAGE_STORE

        # Clear the store for testing
        THREAD_MESSAGE_STORE.clear()

        # Create a message and state
        message = HumanMessage(content="Test message")
        state = MessagesState(messages=[message], thread_id="test-thread")

        # Test update_thread_store
        state.update_thread_store()

        # Verify the message was stored
        assert "test-thread" in THREAD_MESSAGE_STORE
        assert len(THREAD_MESSAGE_STORE["test-thread"]) == 1
        assert THREAD_MESSAGE_STORE["test-thread"][0].content == "Test message"

    def test_from_thread_classmethod(self):
        """Test the from_thread classmethod."""
        from graph.models import MessagesState, HumanMessage, THREAD_MESSAGE_STORE

        # Clear the store for testing
        THREAD_MESSAGE_STORE.clear()

        # Add some messages to the store
        messages = [
            HumanMessage(content="Message 1"),
            HumanMessage(content="Message 2"),
        ]
        THREAD_MESSAGE_STORE["test-thread"] = messages

        # Create a new message
        new_message = HumanMessage(content="New message")

        # Use from_thread to create state
        state = MessagesState.from_thread("test-thread", new_message)

        # Verify the state contains all messages
        assert len(state.messages) == 3
        assert state.thread_id == "test-thread"
        assert state.messages[0].content == "Message 1"
        assert state.messages[1].content == "Message 2"
        assert state.messages[2].content == "New message"

    def test_streaming_response_model(self):
        """Test that StreamingResponse works correctly."""
        from graph.models import StreamingResponse

        # Test content response
        content_response = StreamingResponse(content="Test content", type="content")
        assert content_response.content == "Test content"
        assert content_response.type == "content"

        # Test function call response
        func_response = StreamingResponse(
            type="function_call",
            function_name="test_func",
            function_args={"arg1": "value1"},
        )
        assert func_response.type == "function_call"
        assert func_response.function_name == "test_func"
        assert func_response.function_args["arg1"] == "value1"

    def test_retrieval_result_model(self):
        """Test that RetrievalResult works correctly."""
        from graph.models import RetrievalResult

        result = RetrievalResult(
            content="Test content", score=0.95, metadata={"source": "test"}
        )

        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.metadata["source"] == "test"

    @pytest.mark.asyncio
    async def test_async_functions_exist(self):
        """Test that async functions can be imported and called."""
        from graph.nodes import init_example_selector

        # Test that the function exists and is callable
        assert callable(init_example_selector)

        # Note: We don't actually call it here since it requires
        # external dependencies, but we verify it's importable

    def test_graph_assembly(self):
        """Test that the graph assembly works correctly."""
        from graph.nodes import streaming_graph

        # Test that the graph has the expected structure
        assert streaming_graph.entry_point == "relevance_check"
        assert "relevance_check" in streaming_graph.nodes
        assert "query_or_respond" in streaming_graph.nodes
        assert "few_shot_selector" in streaming_graph.nodes
        assert "generate_with_context" in streaming_graph.nodes
        assert "generate_with_persona" in streaming_graph.nodes

        # Test that edges are properly configured
        assert "relevance_check" in streaming_graph.edges
        assert "query_or_respond" in streaming_graph.edges
        assert "few_shot_selector" in streaming_graph.edges

    def test_utility_functions(self):
        """Test that utility functions can be imported."""
        from graph.nodes import (
            relevance_condition,
            query_or_respond_condition,
            stream_chat_completion,
        )

        # Test that functions exist and are callable
        assert callable(relevance_condition)
        assert callable(query_or_respond_condition)
        assert callable(stream_chat_completion)

    def test_prompt_templates_exist(self):
        """Test that prompt templates file exists and is readable."""
        import json

        template_path = "graph/prompt_templates.json"
        assert os.path.exists(template_path)

        # Test that it's valid JSON
        with open(template_path, "r") as f:
            templates = json.load(f)

        # Test that it has expected structure
        assert isinstance(templates, dict)
        # Add more specific checks based on your template structure


@pytest.mark.asyncio
async def test_relevance_check_node_history_formatting():
    node = RelevanceCheckNode(name="relevance_check")
    # Compose a state with human, ai, and tool messages
    state = MessagesState(
        messages=[
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            ToolMessage(
                content="",
                tool_name="retrieve",
                input={"query": "foo"},
                output=[RetrievalResult(content="Relevant doc", score=0.95)],
            ),
            HumanMessage(content="Is this about Raghu?"),
        ],
        thread_id="test-thread",
    )
    # Patch OpenAI client
    with patch("graph.nodes.client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(
            return_value=type(
                "Resp",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {"message": type("Msg", (), {"content": "CONTEXTUAL"})()},
                        )
                    ]
                },
            )()
        )
        result = await node.process(state)
        # The last message should be an AIMessage with content "CONTEXTUAL"
        assert isinstance(result.messages[-1], AIMessage)
        assert result.messages[-1].content == "CONTEXTUAL"
        # The conversation history should include the retrieved content as assistant message
        openai_messages, _ = node._build_conversation_history(state)
        assert any(
            m["content"].startswith("Content: Relevant doc")
            for m in openai_messages
            if m["role"] == "assistant"
        )
        # Tool call details should not be present
        assert all("tool_call_id" not in m for m in openai_messages)


@pytest.mark.asyncio
async def test_query_or_respond_node_history_formatting():
    node = QueryOrRespondNode(name="query_or_respond")
    state = MessagesState(
        messages=[
            HumanMessage(content="Hi"),
            AIMessage(content="Hello!"),
            ToolMessage(
                content="",
                tool_name="retrieve",
                input={"query": "bar"},
                output=[RetrievalResult(content="Some info", score=0.88)],
            ),
            HumanMessage(content="Do you have enough info?"),
        ],
        thread_id="test-thread",
    )
    # Patch OpenAI client
    with patch("graph.nodes.client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(
            return_value=type(
                "Resp",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {"message": type("Msg", (), {"content": "SUFFICIENT"})()},
                        )
                    ]
                },
            )()
        )
        # Patch the rest of the process logic to only test message formatting and output
        with patch.object(
            QueryOrRespondNode, "process", wraps=node.process
        ) as _wrapped:
            result = await node.process(state)
            assert isinstance(result.messages[-1], AIMessage) or isinstance(
                result.messages[-1], ToolMessage
            )
            openai_messages, last_human_message = node._build_conversation_history(
                state
            )
            assert any(
                m["content"].startswith("Content: Some info")
                for m in openai_messages
                if m["role"] == "assistant"
            )
            assert all("tool_call_id" not in m for m in openai_messages)
            assert last_human_message == "Do you have enough info?"

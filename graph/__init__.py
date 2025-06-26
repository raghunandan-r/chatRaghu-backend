# Graph package initialization
# This provides backward compatibility for existing imports from graph.graph

from .models import (
    MessagesState,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
    StreamingState,
    StreamingResponse,
    Tool,
    RetrievalResult,
    THREAD_MESSAGE_STORE,
    EXAMPLE_EMBEDDINGS,
    QUERY_EMBEDDINGS_CACHE,
)

from .infrastructure import (
    Node,
    StreamingNode,
    StateGraph,
    StreamingStateGraph,
    ClassificationNodeMixin,
    RetrievalNodeMixin,
    SystemPromptNodeMixin,
)

from .retrieval import (
    VectorStore,
    RetrieveTool,
    ExampleSelector,
    vector_store,
    preprocess_text,
    client,
    embedding_client,
    pc,
)

from .nodes import (
    RelevanceCheckNode,
    QueryOrRespondNode,
    FewShotSelectorNode,
    GenerateWithRetrievedContextNode,
    GenerateWithPersonaNode,
    relevance_condition,
    query_or_respond_condition,
    stream_chat_completion,
    streaming_graph,
    set_queue_manager,
    init_example_selector,
)

# For backward compatibility, also export everything from the main modules
__all__ = [
    # Models
    "MessagesState",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "ToolMessage",
    "BaseMessage",
    "StreamingState",
    "StreamingResponse",
    "Tool",
    "RetrievalResult",
    "THREAD_MESSAGE_STORE",
    "EXAMPLE_EMBEDDINGS",
    "QUERY_EMBEDDINGS_CACHE",
    # Infrastructure
    "Node",
    "StreamingNode",
    "StateGraph",
    "StreamingStateGraph",
    "ClassificationNodeMixin",
    "RetrievalNodeMixin",
    "SystemPromptNodeMixin",
    # Retrieval
    "VectorStore",
    "RetrieveTool",
    "ExampleSelector",
    "vector_store",
    "preprocess_text",
    "client",
    "embedding_client",
    "pc",
    # Nodes
    "RelevanceCheckNode",
    "QueryOrRespondNode",
    "FewShotSelectorNode",
    "GenerateWithRetrievedContextNode",
    "GenerateWithPersonaNode",
    "relevance_condition",
    "query_or_respond_condition",
    "stream_chat_completion",
    "streaming_graph",
    "set_queue_manager",
    "init_example_selector",
]

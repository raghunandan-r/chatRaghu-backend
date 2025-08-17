# Graph package initialization
# This provides backward compatibility for existing imports from graph.graph

from .models import (
    MessagesState,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
    Tool,
    RetrievalResult,
    THREAD_MESSAGE_STORE,
    EXAMPLE_EMBEDDINGS,
    QUERY_EMBEDDINGS_CACHE,
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

from .adapters import (
    RelevanceCheckAdapter,
    QueryOrRespondAdapter,
    DeflectionCategorizerAdapter,
    GenerateAnswerAdapter,
)

# Graph assembly helpers for engine
from .assembly import build_graph, create_engine

# For backward compatibility, also export everything from the main modules
__all__ = [
    # Models
    "MessagesState",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "ToolMessage",
    "BaseMessage",
    "Tool",
    "RetrievalResult",
    "THREAD_MESSAGE_STORE",
    "EXAMPLE_EMBEDDINGS",
    "QUERY_EMBEDDINGS_CACHE",
    # Retrieval
    "VectorStore",
    "RetrieveTool",
    "ExampleSelector",
    "vector_store",
    "preprocess_text",
    "client",
    "embedding_client",
    "pc",
    # Adapters
    "RelevanceCheckAdapter",
    "QueryOrRespondAdapter",
    "DeflectionCategorizerAdapter",
    "GenerateAnswerAdapter",
    # Engine assembly
    "build_graph",
    "create_engine",
]

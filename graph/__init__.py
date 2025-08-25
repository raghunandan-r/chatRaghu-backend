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
    vector_store,
    preprocess_text,
)

from .adapters import (
    RouterAdapter,
    GenerateAnswerWithHistoryAdapter,
    GenerateAnswerWithRagAdapter,
    GenerateSimpleResponseAdapter,
)

# Graph assembly helpers for engine
from .assembly import build_graph, create_engine


# Evaluation client components
from .evaluation_queue_manager import EvaluationQueueManager
from .evaluation_client import EvaluationClient, get_evaluation_client, close_evaluation_client
from .evaluation_models import ConversationFlow, EnrichedNodeExecutionLog

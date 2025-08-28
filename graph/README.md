# Graph Package Structure

This directory contains the simplified graph-based conversational engine built on an adapter pattern.

## File Structure

```
graph/
├── __init__.py               # Package entry: exports engine creation and models
├── adapters.py               # Router and generator adapters
├── assembly.py               # Nodes/edges definition and engine factory
├── config.py                 # Graph configuration
├── engine.py                 # GraphEngine orchestration and auditing
├── evaluation_client.py      # Client lifecycle helpers
├── evaluation_models.py      # ConversationFlow, logs, schemas for audits
├── evaluation_queue_manager.py# Queue manager interface
├── models.py                 # Core data models and message types
├── retrieval.py              # Vector store and retrieval tools
├── schemas.py                # Pydantic response models (RoutingDecision, GenerationResponse)
├── utils.py                  # Prompt rendering and helpers
└── README.md                 # This file
```

## Module Overview

### `adapters.py`
Implements the graph nodes as light-weight adapters:
- **RouterAdapter**: Non-streaming decision node producing `RoutingDecision`
- **GenerateSimpleResponseAdapter**: Streaming generator for `greeting`/`deflect`
- **GenerateAnswerWithHistoryAdapter**: Streaming generator using conversation history
- **GenerateAnswerWithRagAdapter**: Streaming generator with retrieval side-effect

### `assembly.py`
Defines the graph topology matching the current design:

```text
nodes:
  router, generate_simple_response, generate_answer_with_history, generate_answer_with_rag

edges:
  router:
    greeting -> generate_simple_response
    deflect -> generate_simple_response
    answer_with_history -> generate_answer_with_history
    retrieve_and_answer -> generate_answer_with_rag
  generate_simple_response -> END
  generate_answer_with_history -> END
  generate_answer_with_rag -> END

entry_point: router
```

### `engine.py`
Coordinates prompt building, LLM calls (streaming/non-streaming), routing, auditing, and enqueueing.

### `schemas.py`
Contains Pydantic models used for structured validation:
- **RoutingDecision** with `decision` and optional `query_for_retrieval`
- **GenerationResponse** with `text`

### `retrieval.py`
`RetrieveTool` integrates vector search used by `GenerateAnswerWithRagAdapter`.

## Usage

### Basic Import
```python
from graph import (
    MessagesState,
    HumanMessage,
    create_engine,
    get_evaluation_client,
    close_evaluation_client,
)
```

### Creating a Conversation State
```python
from graph import MessagesState, HumanMessage

state = MessagesState(
    messages=[HumanMessage(content="Hello, how are you?")],
    thread_id="user-123",
    user_query="Hello, how are you?",
)
```


## Migration Notes

The package maintains compatibility via `graph.__init__` exports. Prefer the new `create_engine` entry point over historical graph builders.

## Benefits of the New Structure

1. **Clear Separation of Concerns**: Each file has a single, focused responsibility
2. **Better Maintainability**: Related functionality is co-located
3. **Easier Testing**: Each component can be tested in isolation
4. **Improved Readability**: Logical organization makes the codebase easier to understand
5. **Scalability**: Easy to add new nodes or retrieval methods
6. **Backward Compatibility**: Existing code continues to work without changes

## Metrics and Evaluation Flow

The engine captures detailed metrics for each turn and passes them to the evaluation service.

### Captured Metrics
- **IDs**: `run_id`, `thread_id`, `turn_index`
- **Latency**: `latency_ms`, `time_to_first_token_ms`
- **Tokens**: Per-node `prompt_tokens`, `completion_tokens`, and aggregated totals

### Data Flow
1. **API (`app.py`)**: Provides `run_id`, `turn_index` and initializes `MessagesState`.
2. **Engine (`engine.py`)**: Orchestrates adapter calls, logs usage, and routes edges.
3. **Queue**: Final `ConversationFlow` is enqueued for the evaluation service.

## Notes

The previous `nodes.py` and `infrastructure.py` are no longer used; functionality has been consolidated into `adapters.py`, `assembly.py`, and `engine.py`.

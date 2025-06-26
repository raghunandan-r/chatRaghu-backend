# Graph Package Structure

This directory contains the refactored graph-based conversational AI system, organized into logical modules for better maintainability and understanding.

## File Structure

```
graph/
├── __init__.py              # Package initialization and exports
├── models.py                # Core data models and message types
├── infrastructure.py        # Graph framework and base classes
├── retrieval.py             # Vector store and retrieval tools
├── nodes.py                 # Processing nodes and graph assembly
├── prompt_templates.json    # Prompt templates for the system
└── README.md               # This file
```

## Module Overview

### `models.py` - Core Data Models
Contains all the data structures used throughout the system:
- **Message Models**: `BaseMessage`, `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`
- **State Models**: `MessagesState`, `StreamingState`, `StreamingResponse`
- **Tool Models**: `Tool`, `RetrievalResult`
- **Global State**: Thread message store and embeddings cache

### `infrastructure.py` - Graph Framework
Provides the foundational classes for the graph system:
- **Base Classes**: `Node`, `StreamingNode`, `StateGraph`, `StreamingStateGraph`
- **Mixins**: `ClassificationNodeMixin`, `RetrievalNodeMixin`, `SystemPromptNodeMixin`
- **Abstract Methods**: Base implementations for node processing and metadata collection

### `retrieval.py` - Retrieval System
Handles all vector search and document retrieval functionality:
- **VectorStore**: Pinecone-based vector database interface
- **RetrieveTool**: Tool for retrieving relevant documents
- **ExampleSelector**: Few-shot learning example selection
- **Utilities**: Text preprocessing and embedding generation

### `nodes.py` - Processing Nodes
Contains all the processing nodes and graph assembly:
- **Processing Nodes**: `RelevanceCheckNode`, `QueryOrRespondNode`, `FewShotSelectorNode`, `GenerateWithRetrievedContextNode`, `GenerateWithPersonaNode`
- **Utility Functions**: Routing conditions and streaming utilities
- **Graph Assembly**: Complete graph configuration and initialization

## Usage

### Basic Import
```python
from graph import (
    MessagesState,
    HumanMessage,
    streaming_graph,
    set_queue_manager
)
```

### Creating a Conversation State
```python
from graph import MessagesState, HumanMessage

message = HumanMessage(content="Hello, how are you?")
state = MessagesState(messages=[message], thread_id="user-123")
```

### Running the Graph
```python
from graph import streaming_graph

# Execute the graph with streaming
async for chunk, metadata in streaming_graph.execute_stream(state):
    print(chunk.content)
```

## Testing

The new structure is tested with pytest. Run the tests with:

```bash
# Run all tests
pytest

# Run only graph structure tests
pytest -m graph_structure

# Run specific test file
pytest tests/test_graph_structure.py
```

## Migration Notes

This structure maintains full backward compatibility with the previous `graph.graph` imports. The `__init__.py` file exports all the necessary components to ensure existing code continues to work without changes.

### Old Import Pattern (Still Works)
```python
from graph.graph import MessagesState, HumanMessage, streaming_graph
```

### New Import Pattern (Recommended)
```python
from graph import MessagesState, HumanMessage, streaming_graph
```

## Benefits of the New Structure

1. **Clear Separation of Concerns**: Each file has a single, focused responsibility
2. **Better Maintainability**: Related functionality is co-located
3. **Easier Testing**: Each component can be tested in isolation
4. **Improved Readability**: Logical organization makes the codebase easier to understand
5. **Scalability**: Easy to add new nodes or retrieval methods
6. **Backward Compatibility**: Existing code continues to work without changes

## File Sizes

- `models.py`: ~90 lines - Pure data structures
- `infrastructure.py`: ~174 lines - Framework and base classes
- `retrieval.py`: ~214 lines - Vector store and retrieval
- `nodes.py`: ~1339 lines - Processing nodes and assembly
- `__init__.py`: ~104 lines - Package exports

Each file is focused and manageable, with clear boundaries between different types of functionality.

# ChatRaghu Backend

A microservices-based backend for raghu.fyi application, consisting of a main API service and a separate evaluation service.

## Architecture

The application is split into two main services:

1. **Main ChatRaghu Backend** (Port 3000): Handles chat requests, document retrieval, and conversation flow
2. **Evaluation Service** (Port 8001): Processes conversation evaluations and quality metrics

## Conversation Flow Architecture

The main service implements a sophisticated graph-based conversation engine with multiple decision nodes:

```mermaid
graph TD
    A[User Query] --> B[Relevance Check]
    B -->|RELEVANT| C[Query or Respond]
    B -->|IRRELEVANT| D[Deflection Categorizer]

    C -->|RETRIEVE| E[Document Retrieval]
    C -->|SUFFICIENT| F[Generate Answer]

    E --> F
    D --> F

    F --> G[Stream Response]
    G --> H[Enqueue for Evaluation]

    H --> I[Message Queue]
    I --> J[Evaluation Service]
    J --> K[Multiple Evaluators]
    K --> L[Store Results]

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#fff3e0
    style F fill:#e8f5e8
    style G fill:#e8f5e8
    style H fill:#f3e5f5
    style I fill:#f3e5f5
    style J fill:#fff8e1
    style K fill:#fff8e1
    style L fill:#fff8e1
```

### Graph Node Details

**Decision Nodes (Non-Streaming):**
- **Relevance Check**: Classifies queries as RELEVANT/IRRELEVANT using structured LLM validation
- **Query or Respond**: Decides between RETRIEVE (for new information) or SUFFICIENT (use conversation history)
- **Deflection Categorizer**: For irrelevant queries, categorizes as OFFICIAL/JEST/HACK

**Generation Node (Streaming):**
- **Generate Answer**: Streams the final response based on context mode (RAG, history, or deflection)

**Context Modes:**
- **RAG**: Uses retrieved documents for context
- **History**: Uses conversation history only
- **Deflection**: Uses deflection category for appropriate response

## Services

### Main Service (`chatraghu-backend`)
- FastAPI application handling chat requests
- **Graph Engine**: Orchestrates conversation flow through decision nodes
- **Adapters**: Modular components for each graph node (relevance, routing, generation)
- **Document Retrieval**: RAG functionality with context-aware retrieval
- **Message Queue**: Asynchronous evaluation enqueueing
- Authentication and rate limiting

### Evaluation Service (`evaluation-service`)
- Separate FastAPI service for evaluation processing
- **Message Queue Integration**: Receives conversation flows via HTTP from main service
- **Modular Evaluators**: Extensible evaluator architecture in `evals-service/evaluators` package
- **Multiple Evaluations**: Per-graph-node evaluations (relevance, persona, etc.)
- Results storage and metrics
- **Structured LLM Validation**: Using Instructor and Pydantic for hallucination prevention

## Service Communication

### Message Queue System

The main service communicates with the evaluation service using an **asynchronous message queue**:

1. **Queue Manager**: `EvaluationQueueManager` maintains an in-memory `asyncio.Queue`
2. **Background Worker**: Processes evaluation requests in the background
3. **HTTP Communication**: Sends `ConversationFlow` objects to evaluation service via HTTP
4. **Retry Logic**: Handles failures with automatic retries
5. **Graceful Shutdown**: Clean cancellation on service shutdown

**Queue Flow:**
```
Main Service → Queue Manager → Background Worker → HTTP Client → Evaluation Service
```

### API Endpoints

- **Asynchronous Evaluation**: Main service sends evaluation requests to `/evaluate`
- **Health Monitoring**: Regular health checks via `/health`
- **Queue Management**: Internal queue operations for conversation flow processing

## Recent Updates

### Graph Engine Enhancements

**Modular Adapter Architecture**
- **NEW**: Each graph node implemented as a separate adapter class
- **NEW**: Structured decision schemas with Pydantic validation
- **NEW**: Streaming and non-streaming node support
- **NEW**: Context-aware routing based on LLM decisions

**Enhanced Conversation Flow**
- **NEW**: Multi-stage decision pipeline (relevance → routing → generation)
- **NEW**: Dynamic context mode selection (RAG, history, deflection)
- **NEW**: Comprehensive audit logging with token usage tracking
- **NEW**: Opik integration for distributed tracing

### Evaluation Service Enhancements

**Structured LLM Validation with Instructor**
- **NEW**: Replaced manual JSON parsing with Instructor-patched OpenAI client
- **NEW**: Added `LLMRelevanceJudgement` Pydantic model for strict response validation
- **NEW**: Eliminated LLM hallucination through structured validation
- **NEW**: Enhanced error handling and logging for evaluation processes

**Message Queue Integration**
- **NEW**: Asynchronous conversation flow processing
- **NEW**: Background worker with graceful shutdown
- **NEW**: HTTP-based communication between services
- **NEW**: Comprehensive error handling and retry logic

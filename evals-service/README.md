# ChatRaghu Evaluation Service

This is a separate FastAPI service that handles conversation evaluation for the ChatRaghu backend. It processes conversation flows and evaluates the quality of responses using various metrics.

## Features

- **Asynchronous Evaluation**: Processes evaluation requests in the background
- **Synchronous Evaluation**: Provides immediate evaluation results for testing
- **Multiple Evaluation Types**:
  - Relevance check evaluation
  - Query/respond decision evaluation
  - Few-shot selector evaluation
  - Context generation evaluation
  - Persona consistency evaluation
- **Queue Management**: Handles evaluation requests with retry logic
- **Health Monitoring**: Provides health check and metrics endpoints

## API Endpoints

### POST `/evaluate`
Submit a conversation for asynchronous evaluation.

**Request Body:**
```json
{
  "thread_id": "string",
  "query": "string",
  "response": "string",
  "retrieved_docs": [{"page_content": "string", "metadata": {}}],
  "conversation_flow": {
    "thread_id": "string",
    "user_query": "string",
    "node_executions": [...]
  }
}
```

### POST `/evaluate/sync`
Submit a conversation for synchronous evaluation (returns results immediately).

### GET `/health`
Health check endpoint.

### GET `/metrics`
Get service metrics including queue size and worker status.

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for evaluation
- `PINECONE_API_KEY`: Pinecone API key (if needed)
- `SENTRY_DSN`: Sentry DSN for error tracking (optional)

## Running the Service

### Development
```bash
cd evals-service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### Docker
```bash
docker build -t chatraghu-evaluation .
docker run -p 8001:8001 --env-file .env chatraghu-evaluation
```

### Docker Compose
```bash
docker-compose up evaluation-service
```

## Integration with Main Service

The main ChatRaghu backend service communicates with this evaluation service using the `evaluation_client.py` module. The client handles:

- HTTP communication with the evaluation service
- Request/response serialization
- Error handling and retries
- Connection pooling

## Evaluation Results

Evaluation results are stored in the `eval_results` directory as Parquet files. Each evaluation includes:

- Thread ID and timestamp
- Query and response
- Retrieved documents
- Node-by-node evaluation scores
- Overall success metrics
- Detailed explanations for each evaluation

## Architecture

```
Main Service (Port 3000)     Evaluation Service (Port 8001)
     |                              |
     |-- HTTP Request ------------>|
     |                              |-- Process Evaluation
     |                              |-- Store Results
     |<-- Success Response --------|
```

The evaluation service operates independently and can be scaled separately from the main service.
